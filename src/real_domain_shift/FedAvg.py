import os 
import random
import tracemalloc
import math
import numpy as np
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import argparse
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder, FashionMNIST
from torchvision.transforms.functional import rotate
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from PIL import Image, ImageFile
import torchvision.datasets.folder
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser() 
parser.add_argument('--dataset', type=str, default='offcie_caltech10', help="name \of dataset") 
parser.add_argument('--data_root', type=str, default='./data', help='root directory containing the datasets (PACS, VLCS, Office_Caltech10, OfficeHome)')
parser.add_argument('--batch_sizeS', type=int, default=32, help='batch size') #Fixed
parser.add_argument('--batch_sizeT', type=int, default=8, help='batch size') #Fixed
parser.add_argument('--global_epoch', type=int, default=50, help='local epoch') #Fixed
parser.add_argument('--seed', type=int, default=50, help='random seed') ##Fixed

parser.add_argument('--learning_rate', type=float, default=0.01 , help='choose learning rate for optimizer')
parser.add_argument('--learning_rate_target', type=float, default=0.001 , help='choose learning rate for optimizer')

parser.add_argument('--device_server', type=int, default=1, help='gpu number for server')
parser.add_argument('--device_local', type=int, default=4, help='gpu number for server')

args = parser.parse_args()
np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False
torch.manual_seed(args.seed) 
random.seed(args.seed)
ImageFile.LOAD_TRUNCATED_IMAGES = True
#############################################################################################     
def flat_grad(w_list): #checked
    g_list=[]
    for param in w_list:
        g_list.append(torch.flatten(param))
    return torch.cat(g_list)

def get_param(model): #checked
    #with torch.no_grad():
    params=[]
    for param in model.parameters():
        params.append(param.detach().clone())
    return params

def makeweights_list(cnn1, weights_tensor): #checked
    cnn=copy.deepcopy(cnn1)
    #with torch.no_grad():
    lis=[]
    param_init_indx=0
    for param in cnn.parameters():
        p=param.detach().clone()
        shpe=p.shape
        param_size=p.reshape(-1).shape[0]
        replace_eles=weights_tensor[param_init_indx:param_init_indx+param_size]
        param_init_indx+=param_size
        lis.append(replace_eles.reshape(shpe))     
    del cnn
    return lis
#######################################################################################################################    
def evaluate_model(test_dataset, model, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    all_probs = []

    # Create a DataLoader from the test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_sizeS, shuffle=False)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)  # Get probabilities
            preds = torch.argmax(outputs, dim=1)    # Get predicted classes

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Concatenate all predictions, labels, and probabilities
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Calculate AUC for multi-class
    n_classes = all_probs.shape[1]  # Number of classes
    auc = roc_auc_score(np.eye(n_classes)[all_labels], all_probs, multi_class='ovr')

    # Calculate Cohen's Kappa Score
    kappa = cohen_kappa_score(all_labels, all_preds)

    return auc, accuracy, kappa

def average_weights(w, device):
    w_avg = copy.deepcopy(w[0]).to(device)
    w_avg=w_avg.state_dict()
    #print(w_avg)
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i].state_dict()[key].to(device)
        w_avg[key] = w_avg[key]/len(w)
    return w_avg
###############################################################################################################

def local_update(device, cnn, train_set, lr, bz):
    model=copy.deepcopy(cnn)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_loader=torch.utils.data.DataLoader(train_set, batch_size=bz, shuffle=True, num_workers=2)
    for batch_idx, (images, labels) in enumerate(train_loader):
        model.train()
        #images = images.to(device)
        #labels = labels.to(device)
        loss = criterion(model(images.to(device)), labels.to(device)) 
        model.zero_grad()
        loss.backward() 
        g = flat_grad([para.grad.detach().clone() for para in model.parameters()]) 
        g=g*lr
        g = flat_grad(get_param(model)) - g
        g=makeweights_list(model, g)        
        for index, param in enumerate(model.parameters()):
            param.data=g[index].clone() 
    del train_loader
    return model
################################################################################################
##Split target dataset 2% ssssssssss
if __name__ == '__main__':
    print("preparing data")
    
    if args.dataset == 'PACS':  ####1670, 2048, 2344, 3929
        img_size=224
        n_channel=3
        n_classes=7
        num_clients = 4
        # Root directory where PACS dataset is stored
        root = os.path.join(args.data_root, "PACS", "kfold")  # PACS dataset root
        
        # Define domain names for PACS dataset
        domains = ['photo', 'art_painting', 'cartoon', 'sketch']  # 'sketch' is the target domain
        
        # Define target domain (you can change this to experiment with different target domains)
        target_domain = 'sketch'  # Set 'sketch' as the target domain
        
        # Define basic transformations (for the target domain)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Define augmentation transformations for source domains
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Initialize lists for source and target datasets
        sub_sources = []
        target_data = None
        
        # Load the datasets for each domain
        for domain in domains:
            domain_path = os.path.join(root, domain)  # Path for each domain
            if domain == target_domain:  # Target domain (no augmentation)
                dataset = ImageFolder(domain_path, transform=transform)
                target_data = dataset
            else:  # Source domains (apply augmentation)
                dataset = ImageFolder(domain_path, transform=augment_transform)
                sub_sources.append(dataset)
        percentge = 2/100
        # Get the length of the dataset
        dataset_size = len(target_data) 
        # Define the split sizes
        train_size = int(percentge * dataset_size)
        test_size = dataset_size - train_size
        # Use random_split to split the dataset
        target_train, target_test = torch.utils.data.random_split(target_data, [train_size, test_size])    
        del target_data
        cnn = models.resnet18(pretrained=False)
        num_ftrs = cnn.fc.in_features
        cnn.fc = torch.nn.Linear(num_ftrs, 7)        
   
    elif args.dataset == 'VLCS': ###1415, 2656, 3282, 3376
        img_size=224
        n_channel=3
        n_classes=5
        num_clients = 4
        # env = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
        # Define the root directory for the VLCS dataset
        root = os.path.join(args.data_root, "VLCS")  # VLCS dataset root
        hparams = {'data_augmentation': True}  # Hyperparameters (e.g., for augmentation)
        # List of environments (folders inside the root directory)
        environments = sorted([f.name for f in os.scandir(root) if f.is_dir()])

        # Define the common transformations (resize, normalization)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define the augmentation transformations (random crop, flip, color jitter)
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_env_index = 3
        sub_sources = []
        target_data = None
        
        # Loop through all environments to separate source and test datasets
        for i, env in enumerate(environments):
            if i == test_env_index:
                # This is the test environment, apply the basic transform
                test_transform = transform
                target_data = ImageFolder(os.path.join(root, env), transform=test_transform)
            else:
                # These are source environments, apply augmentation transform
                source_transform = augment_transform
                #source_transform = augment_transform if hparams['data_augmentation'] else transform
                source_dataset = ImageFolder(os.path.join(root, env), transform=source_transform)
                sub_sources.append(source_dataset)
        percentge = 2/100
        # Get the length of the dataset
        dataset_size = len(target_data)
        #print(dataset_size)       
        # Define the split sizes
        train_size = int(percentge * dataset_size)
        test_size = dataset_size - train_size
        
        # Use random_split to split the dataset
        target_train, target_test = torch.utils.data.random_split(target_data, [train_size, test_size])    
        del target_data
        #Resnet18
        cnn = models.resnet18(pretrained=False)
        num_ftrs = cnn.fc.in_features
        cnn.fc = torch.nn.Linear(num_ftrs, 5) 
               
      
    elif args.dataset == 'OfficeHome': 
        img_size=224
        n_channel=3
        n_classes=65
        num_clients = 4
        # Root directory where PACS dataset is stored
        # Root directory where Office-Home dataset is stored
        root = os.path.join(args.data_root, "OfficeHome", "OfficeHomeDataset_10072016")  # Office-Home dataset root
        
        # Define domain names for Office-Home dataset
        domains = ['Art', 'Clipart', 'Product', 'Real World']  # 'Art' is the target domain
        
        # Define target domain (you can change this to experiment with different target domains)
        target_domain = 'Art'  # Set 'Art' as the target domain
        
        # Define basic transformations (for the target domain)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])        
        '''# Define augmentation transformations for source domains
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])'''
        
        # Initialize lists for source and target datasets
        sub_sources = []
        target_data = None
        
        # Load the datasets for each domain
        for domain in domains:
            domain_path = os.path.join(root, domain)  # Path for each domain
            if domain == target_domain:  # Target domain (no augmentation)
                dataset = ImageFolder(domain_path, transform=transform)
                target_data = dataset
            else:  # Source domains (apply augmentation)
                dataset = ImageFolder(domain_path, transform=augment_transform)
                sub_sources.append(dataset)
                
        percentge = 2/100
        # Get the length of the dataset
        dataset_size = len(target_data)     
        # Define the split sizes
        train_size = int(percentge * dataset_size)
        test_size = dataset_size - train_size
        # Use random_split to split the dataset
        target_train, target_test = torch.utils.data.random_split(target_data, [train_size, test_size])    
        del target_data
        #Resnet18
        cnn = models.resnet18(pretrained=False)
        num_ftrs = cnn.fc.in_features
        cnn.fc = torch.nn.Linear(num_ftrs, 65) 
        
    elif args.dataset == 'offcie_caltech10':  
        img_size=224
        n_channel=3
        n_classes=10
        num_clients = 4
        # Define the transform to normalize and resize the images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to a consistent size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Pre-trained model normalization
        ])
        ###env = [amazon, caltech10, dslr, webcam] 958 1123 157 295
        # Paths to the directories for each domain
        amazon_dir = os.path.join(args.data_root, "Office_Caltech10", "amazon")
        caltech_dir = os.path.join(args.data_root, "Office_Caltech10", "caltech10")
        dslr_dir = os.path.join(args.data_root, "Office_Caltech10", "dslr")
        webcam_dir = os.path.join(args.data_root, "Office_Caltech10", "webcam")
        data_list = []
        # Create datasets using ImageFolder for each domain
        data = ImageFolder(root=amazon_dir, transform=transform)
        data_list.append(data)
        data = ImageFolder(root=caltech_dir, transform=transform)
        data_list.append(data)
        data = ImageFolder(root=dslr_dir, transform=transform)
        data_list.append(data)
        data = ImageFolder(root=webcam_dir, transform=transform)
        data_list.append(data)
        print(len(data_list[0]), len(data_list[1]), len(data_list[2]), len(data_list[3]))
        del data
        # Example: Loading one batch from each domain
        p = [0, 1, 2, 3]
        Target_index = 3
        target_data = data_list[Target_index]
        sub_sources = [data_list[i] for i in p if i != Target_index]
        del data_list  
        
        percentge = 20/100
        # Get the length of the dataset
        dataset_size = len(target_data)     
        # Define the split sizes
        train_size = int(percentge * dataset_size)
        test_size = dataset_size - train_size
        # Use random_split to split the dataset
        target_train, target_test = torch.utils.data.random_split(target_data, [train_size, test_size])    
        del target_data
        
        #Resnet18
        cnn = models.resnet18(pretrained=False)
        num_ftrs = cnn.fc.in_features
        cnn.fc = torch.nn.Linear(num_ftrs, 10)       
#############################################################################################
    print("Fed Training")
    cnn.to(args.device_server)
    target_accuracy=[]   
    total_time=0;
    for epoch in range(args.global_epoch):
        start_time = time.time()
        ModTarget = local_update(args.device_local, cnn, target_train, args.learning_rate_target, args.batch_sizeT)
        ModTarget = flat_grad(get_param(ModTarget)).to(args.device_server)
        count = 0
        for i in range(len(sub_sources)): 
            count+=1
            mod = local_update(args.device_local, cnn, sub_sources[i], args.learning_rate, args.batch_sizeS)
            ModTarget+= flat_grad(get_param(mod)).to(args.device_server) 
        ModTarget = ModTarget/(count+1)
        del mod  
        #FedGP aggregation
        ModTarget=makeweights_list(cnn, ModTarget)        
        for index, param in enumerate(cnn.parameters()):
            param.data=ModTarget[index].clone() 
        del ModTarget
        auc, acc, ka = evaluate_model(target_test, cnn, args.device_server)
        target_accuracy.append(acc)
        round_time = time.time() - start_time
        total_time = total_time + round_time
        print("{} ".format(epoch+1), "{:.4f}".format(auc), "{:.4f}".format(acc), "{:.4f}".format(ka), "{:.3f}".format(total_time))
        
    print("Best accuracies in all the target clients:", max(target_accuracy))
