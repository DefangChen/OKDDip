'''
This is PyTorch 1.0 re-implementation for knowledge distillation on CIFAR-10/100 and ImageNet.
'''
import argparse
import logging
import os
import random
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import utils

import models
import models.data_loader as data_loader
from tensorboardX import SummaryWriter

# Fix the random seed for reproducible experiments
# random.seed(97)
# torch.manual_seed(97)
# if torch.cuda.is_available(): torch.cuda.manual_seed(97)
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# Set parameters
parser = argparse.ArgumentParser()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
    
parser.add_argument('--model', metavar='ARCH', default='resnet32', type=str,
                    choices=model_names, help='Student model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')    
parser.add_argument('--T_model', metavar='ARCH', default='resnet110', type=str, 
                    choices=model_names, help='Teacher model architecture: ' + ' | '.join(model_names) + ' (default: resnet110)')
parser.add_argument('--num_workers', default=8, type=int, help = 'Input the number of works: default(0)')
parser.add_argument('--T_model_path', default='',help = 'Decide whether or not to use specified path: default('')')
parser.add_argument('--loss', default='KL', type=str, help = 'Define the loss between student output and teacher output: default(KL_Loss)')
parser.add_argument('--temperature', default=3.0, type=float, help = 'Input the temperature: default(3.0)')
parser.add_argument('--alpha', default=1.0, type=float, help = 'Input the relative weight: default(1.0)')
parser.add_argument('--dataset', default='CIFAR10', type=str, help = 'Input the dataset name: default(CIFAR10)')
parser.add_argument('--num_epochs', default=300, type=int, help = 'Input the number of epoches: default(300)')
parser.add_argument('--batch_size', default=128, type=int, help = 'Input the batch size: default(128)')
parser.add_argument('--lr', default=0.1, type=float, help = 'Input the learning rate: default(0.1)')
parser.add_argument('--efficient', action='store_true', help = 'Decide whether or not to use efficient implementation: default(False)')
parser.add_argument('--wd', default=5e-4, type=float, help = 'Input the weight decay rate: default(5e-4)')
parser.add_argument('--dropout', default=0., type=float, help = 'Input the dropout rate: default(0.0)')
parser.add_argument('--resume', default='', type=str, help = 'Input the path of resume model: default('')')
parser.add_argument('--version', default='V0', type=str, help = 'Input the version of current model: default(V0)')
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(args)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_loader, model, model_T, optimizer, criterion, criterion_T, accuracy, args):
    # set model to training mode
    model.train()
    # set teacher model to evaluation mode
    model_T.eval()
    
    # summary for current training loop and a running average object for loss   
    loss_avg = utils.RunningAverage()
    loss_true_avg = utils.RunningAverage()
    loss_teacher_avg = utils.RunningAverage()
    accTop1_avg = utils.RunningAverage()
    end = time.time()
    
    # Use tqdm for progress bar
    with tqdm(total=len(train_loader)) as t:
        for i, (train_batch, labels_batch) in enumerate(train_loader):
            # move to GPU if available
            train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
            
            # compute model output and loss
            output_batch = model(train_batch)
            with torch.no_grad():
                teacher_outputs = model_T(train_batch)
    
            loss_true = criterion(output_batch, labels_batch)
            loss_teacher = criterion_T(output_batch, teacher_outputs)
            
            loss = loss_true + args.alpha * loss_teacher
            
            # Update average loss and accuracy
            metrics = accuracy(output_batch, labels_batch)
            accTop1_avg.update(metrics[0].item())
            loss_true_avg.update(loss_true.item())
            loss_teacher_avg.update(loss_teacher.item())
            loss_avg.update(loss.item())

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            
            # performs updates using calculated gradients
            optimizer.step()
        
            t.update()
    
    # compute mean of all metrics in summary
    train_metrics = {'train_loss': loss_avg.value(),
                     'train_true_loss': loss_true_avg.value(),
                     'train_teacher_loss': loss_teacher_avg.value(),
                     'train_accTop1': accTop1_avg.value(),
                     'time': time.time() - end}
        
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    logging.info("- Train metrics: " + metrics_string)
    return train_metrics

    
def evaluate(test_loader, model, model_T, criterion, criterion_T, accuracy, args): 
    # set model to evaluation mode
    model.eval()
    # set teacher model to evaluation mode
    model_T.eval()
    
    loss_avg = utils.RunningAverage()
    # loss_teacher_avg = utils.RunningAverage()
    # loss_true_avg = utils.RunningAverage()
    accTop1_avg = utils.RunningAverage()
    end = time.time()

    with torch.no_grad():
        for _, (test_batch, labels_batch) in enumerate(test_loader):
            test_batch, labels_batch = test_batch.to(device), labels_batch.to(device)
             
            # compute model output and loss
            output_batch = model(test_batch)
            
            # loss_true = criterion(output_batch, labels_batch)
            # loss_teacher = criterion_T(output_batch, teacher_outputs)            
            # loss = loss_true + args.alpha * args.temperature * args.temperature * loss_teacher
            loss = criterion(output_batch, labels_batch)
            
            # Update average loss and accuracy
            metrics = accuracy(output_batch, labels_batch)
            accTop1_avg.update(metrics[0].item())
            # loss_true_avg.update(loss_true.item())
            # loss_teacher_avg.update(loss_teacher.item())
            loss_avg.update(loss.item())

    # compute mean of all metrics in summary
    # 'test_true_loss': loss_true_avg.value(),
    # 'test_teacher_loss': loss_teacher_avg.value(),
    test_metrics = { 'test_loss': loss_avg.value(),
                     'test_accTop1': accTop1_avg.value(),
                     'time': time.time() - end}
        
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- Test metrics: " + metrics_string)
    return test_metrics
    
def train_and_evaluate(model, model_T, train_loader, test_loader, optimizer, criterion, criterion_T, accuracy, model_dir, args):
    
    start_epoch = 0
    best_acc = 0.
    
    # learning rate schedulers for different models:
    scheduler = MultiStepLR(optimizer, milestones=[0.5 * args.num_epochs, 0.75 * args.num_epochs], gamma=0.1)
    
    # TensorboardX setup
    writer = SummaryWriter(log_dir = model_dir)
    
    # Save best accTop1
    choose_accTop1 = True
    
    # Save the parameters for export
    result_train_metrics = list(range(args.num_epochs))
    result_test_metrics = list(range(args.num_epochs))

    # If the training is interruptted 
    if args.resume:
        # Load checkpoint.
        logging.info('Resuming from checkpoint..')
        resumePath = os.path.join(args.resume, 'last.pth')
        assert os.path.isfile(resumePath), 'Error: no checkpoint directory found!'                      
        
        checkpoint = torch.load(resumePath)        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])        
        # resume from the last epoch
        start_epoch = checkpoint['epoch']
        scheduler.step(start_epoch - 1)
        if choose_accTop1:
            best_acc = checkpoint['test_accTop1']
        else:
            best_acc = checkpoint['test_accTop5']
        result_train_metrics = torch.load(os.path.join(args.resume, 'train_metrics'))
        result_test_metrics = torch.load(os.path.join(args.resume, 'test_metrics'))
                
    for epoch in range(start_epoch, args.num_epochs):
     
        scheduler.step()
     
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))
            
        # compute number of batches in one epoch (one full pass over the training set)    
        train_metrics = train(train_loader, model, model_T, optimizer, criterion, criterion_T, accuracy, args)
		
        writer.add_scalar('Train/Loss', train_metrics['train_loss'], epoch+1)
        writer.add_scalar('Train/Loss_true', train_metrics['train_true_loss'], epoch+1)
        writer.add_scalar('Train/Loss_teacher', train_metrics['train_teacher_loss'], epoch+1)
        writer.add_scalar('Train/AccTop1', train_metrics['train_accTop1'], epoch+1)
        
		# Evaluate for one epoch on validation set
        test_metrics = evaluate(test_loader, model, model_T, criterion, criterion_T, accuracy, args)        
        
        # Find the best accTop1 for Branch1.
        if choose_accTop1:
            test_acc = test_metrics['test_accTop1']
        else:
            test_acc = test_metrics['test_accTop5']
        
        writer.add_scalar('Test/Loss', test_metrics['test_loss'], epoch+1)
        # writer.add_scalar('Test/Loss_true', test_metrics['test_true_loss'], epoch+1)
        # writer.add_scalar('Test/Loss_teacher', test_metrics['test_teacher_loss'], epoch+1)
        writer.add_scalar('Test/AccTop1', test_metrics['test_accTop1'], epoch+1)
        
        result_train_metrics[epoch] = train_metrics
        result_test_metrics[epoch] = test_metrics

        # Save latest train/test metrics
        torch.save(result_train_metrics, os.path.join(model_dir, 'train_metrics'))
        torch.save(result_test_metrics, os.path.join(model_dir, 'test_metrics'))
        
        last_path = os.path.join(model_dir, 'last.pth')
        # Save latest model weights, optimizer and accuracy
        torch.save({'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'test_accTop1': test_metrics['test_accTop1']}, last_path)
        
        # If best_eval, best_save_path
        is_best = test_acc >= best_acc
        if is_best:
            logging.info("- Found better accuracy")            
            best_acc = test_acc            
            # Save best metrics in a json file in the model directory
            test_metrics['epoch'] = epoch + 1
            utils.save_dict_to_json(test_metrics, os.path.join(model_dir, "test_best_metrics.json"))
        
            # Save model and optimizer
            shutil.copyfile(last_path, os.path.join(model_dir, 'best.pth'))
    writer.close()   

if __name__ == '__main__':

    begin_time = time.time()
    # Set the model directory    
    model_dir= os.path.join('.', args.dataset, '300', 'kd', args.T_model + args.model + args.loss + args.version)
    if not os.path.exists(model_dir):
        print("Directory does not exist! Making directory {}".format(model_dir))
        os.makedirs(model_dir)
    
    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    
    # set number of classes
    if args.dataset == 'CIFAR10':
        num_classes = 10
        model_folder = "model_cifar"
        root='./Data'
    elif args.dataset == 'CIFAR100':
        num_classes = 100
        model_folder = "model_cifar"
        root='./Data'
    elif args.dataset == 'imagenet':
        num_classes = 1000
        model_folder = "model_imagenet"
        root = './Data'
    
    # Load data
    train_loader, test_loader = data_loader.dataloader(data_name = args.dataset, batch_size = args.batch_size, root=root)
    logging.info("- Done.")
    
    # Training from scratch for student model
    model_fd = getattr(models, model_folder)
    if "resnet" in args.model:
        model_cfg = getattr(model_fd, 'resnet')
        model = getattr(model_cfg, args.model)(num_classes = num_classes)
    elif "vgg" in args.model:
        model_cfg = getattr(model_fd, 'vgg')
        model = getattr(model_cfg, args.model)(num_classes = num_classes, dropout = args.dropout)
    elif "densenet" in args.model:
        model_cfg = getattr(model_fd, 'densenet')
        model = getattr(model_cfg, args.model)(num_classes = num_classes)
    
    # Set pretrained teacher path / Born-Again implementation
    if args.T_model_path:
        path_T = os.path.join(args.T_model_path, 'best.pth')
    else: 
        path_T = os.path.join('.', args.dataset, args.T_model, 'best.pth')
        
    # load pretrained teacher model        

    if "resnet" in args.T_model:
        model_cfg = getattr(model_fd, 'resnet')
        model_T = getattr(model_cfg, args.T_model)(pretrained=True, path=path_T, num_classes = num_classes)
    elif "vgg" in args.T_model:
        model_cfg = getattr(model_fd, 'vgg')
        model = getattr(model_cfg, args.T_model)(pretrained=True, path=path_T, num_classes = num_classes, dropout = args.dropout)
    elif "densenet" in args.T_model:
        model_cfg = getattr(model_fd, 'densenet')
        model = getattr(model_cfg, args.T_model)(pretrained=True, path=path_T, num_classes = num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0,1,2,3]).to(device)
        model_T = nn.DataParallel(model_T, device_ids=[0,1,2,3]).to(device)
    else:
        model = model.to(device)
        model_T = model_T.to(device)
        
    num_params = (sum(p.numel() for p in model.parameters())/1000000.0)
    logging.info('Total params: %.2fM' % num_params)
    
    # Loss and optimizer(SGD with 0.9 momentum)
    criterion = nn.CrossEntropyLoss()
    if args.loss == "KL":
        criterion_T = utils.KL_Loss(args.temperature)
    elif args.loss == "CE":
        criterion_T = utils.CE_Loss(args.temperature).to(device)
    accuracy = utils.accuracy
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay = args.wd)    
    
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, model_T, train_loader, test_loader, optimizer, criterion, criterion_T, accuracy, model_dir, args)
    
    logging.info('Total time: {:.2f} hours'.format((time.time() - begin_time)/3600.0))
    state['Total params'] = num_params
    params_json_path = os.path.join(model_dir, "parameters.json") # save parameters
    utils.save_dict_to_json(state, params_json_path)