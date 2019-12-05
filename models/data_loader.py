"""
    mnist, CIFAR-10/100 data normalization:
    
    import numpy as np
    from torchvision import datasets, transforms
    train_transform = transforms.Compose([transforms.ToTensor()])

    # cifar10
    train_set = datasets.CIFAR10(root='../data/', train=True, download=True, transform=train_transform)
    print(train_set.train_data.shape)
    print(train_set.train_data.mean(axis=0).mean(axis=0).mean(axis=0)/255)
    print(train_set.train_data.std(axis=0).mean(axis=0).mean(axis=0)/255)
    # (50000, 32, 32, 3)
    # [0.49139968  0.48215841  0.44653091]
    # [0.24703223  0.24348513  0.26158784]

    # cifar100
    train_set = datasets.CIFAR100(root='../data/', train=True, download=True, transform=train_transform)
    print(train_set.train_data.shape)
    print(train_set.train_data.mean(axis=0).mean(axis=0).mean(axis=0)/255)
    print(train_set.train_data.std(axis=0).mean(axis=0).mean(axis=0)/255)
    # (50000, 32, 32, 3)
    # [0.50707516  0.48654887  0.44091784]
    # [0.26733429  0.25643846  0.27615047]
    
    # mnist
    train_set = datasets.MNIST(root='../data/', train=True, download=True, transform=train_transform)
    print(list(train_set.train_data.size()))
    print(train_set.train_data.float().mean()/255)
    print(train_set.train_data.float().std()/255)
    # [60000, 28, 28]
    # 0.1306604762738429
    # 0.30810780717887876

"""
import os
import torch
import torchvision
import torchvision.transforms as transforms

def dataloader(data_name= "CIFAR100", batch_size= 64, num_workers = 8, root = './Data'):
    """
    Fetch and return train/test dataloader.
    """
    kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': torch.cuda.is_available()}
    
    # normalize all the dataset
    if data_name == "CIFAR10":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    elif data_name == "CIFAR100":
        normalize = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    elif data_name == "imagenet":
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
    if data_name == "CIFAR10" or data_name == "CIFAR100":  
        # Transformer for train set: random crops and horizontal flip
        train_transformer = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
                transforms.ToTensor(),
                normalize])
        # Transformer for test set
        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize])
            
    elif data_name == 'imagenet':
        # Transformer for train set: random crops and horizontal flip
        train_transformer = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
                transforms.ToTensor(),
                normalize])

        # Transformer for test set
        test_transformer = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            
    # Choose corresponding dataset
    if data_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True,
            download=True, transform=train_transformer)
        
        testset = torchvision.datasets.CIFAR10(root=root, train=False,
            download=True, transform=test_transformer)
            
    elif data_name == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True,
            download=True, transform=train_transformer)
        
        testset = torchvision.datasets.CIFAR100(root=root, train=False,
            download=True, transform=test_transformer)
            
    elif data_name == 'imagenet':
        traindir = os.path.join(root, 'train')
        valdir = os.path.join(root, 'val')
        
        trainset = torchvision.datasets.ImageFolder(traindir, train_transformer)
        testset = torchvision.datasets.ImageFolder(valdir, test_transformer)
        # trainset = torchvision.datasets.ImageNet(root=root, split='train', download=False, transform=train_transformer)
        # testset = torchvision.datasets.ImageNet(root=root, split='val', download=False, transform=test_transformer)
        
    trainloader = torch.utils.data.DataLoader(trainset, shuffle = True, **kwargs)
    
    testloader = torch.utils.data.DataLoader(testset, shuffle = False, **kwargs)

    return trainloader, testloader
