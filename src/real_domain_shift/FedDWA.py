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

parser.add_argument('--learning_rate', type=float, default=0.1 , help='choose learning rate for optimizer')
parser.add_argument('--learning_rate_target', type=float, default=0.01, help='choose learning rate for optimizer')

parser.add_argument('--device_server', type=int, default=1, help='gpu number for server')
parser.add_argument('--device_local', type=int, default=7, help='gpu number for server')

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
    
    
    
def average_weights(c_indx, models, P, device):
    w_avg = copy.deepcopy(w[0]).to(device)
    w_avg=w_avg.state_dict()
    #print(w_avg)
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i].state_dict()[key].to(device)
        w_avg[key] = w_avg[key]/len(w)
    return w_avg    

def inverse_square_euclidean_distance(w1, w2):
    # Step 1: Calculate the difference between the two tensors
    diff = w1 - w2
    
    # Step 2: Square each element-wise difference
    squared_diff = diff.pow(2)
    
    # Step 3: Sum all the squared differences
    sum_squared_diff = torch.sum(squared_diff)
    
    # Step 4: Take the square root of the sum
    euclidean_distance = torch.sqrt(sum_squared_diff)
    
    # Step 5: Compute the inverse square of the resulting value
    inverse_square_distance = 1.0 / (euclidean_distance ** 2)
    
    return inverse_square_distance
#####################################################################################################################
def local_update_sources(client_idx, device, train_set, clients_per_models):
    criterion = nn.CrossEntropyLoss()
    train_loader=torch.utils.data.DataLoader(train_set, batch_size=args.batch_sizeS, shuffle=True, num_workers=2)
    # Load i-th client's personalized model and find performance
    model=clients_per_models[client_idx].to(device)    
    ###Find local model
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)   
    running_loss=0
    examples=0
    for batch_idx, (images, labels) in enumerate(train_loader):
        #model.train()
        images = images.to(device)
        labels = labels.to(device)
        #images = torch.autograd.Variable(images.view(-1,input_size)).to(device)
        #labels = torch.autograd.Variable(labels).to(device)
        loss = criterion(model(images), labels) 
        running_loss+=loss.item()*labels.shape[0]
        examples+=labels.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
    model_c = copy.deepcopy(model)
    #Find gaidance model   
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    for batch_idx, (images, labels) in enumerate(train_loader):
        #model.train()

        images = images.to(device)
        labels = labels.to(device)
        #images = torch.autograd.Variable(images.view(-1,input_size)).to(device)
        #labels = torch.autograd.Variable(labels).to(device)
        loss = criterion(model(images), labels) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
    ###Train base of the local model, keep freeze the head
    del train_loader, images, 
    return model, model_c

def local_update_target(device, train_set, testset, clients_per_models):
    criterion = nn.CrossEntropyLoss()
    train_loader=torch.utils.data.DataLoader(train_set, batch_size=args.batch_sizeT, shuffle=True, num_workers=2)
    # Load i-th client's personalized model and find performance
    model=clients_per_models[0].to(device)    
    ###Find local model
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate_target)   
    running_loss=0
    examples=0
    for batch_idx, (images, labels) in enumerate(train_loader):
        #model.train()
        images = images.to(device)
        labels = labels.to(device)
        #images = torch.autograd.Variable(images.view(-1,input_size)).to(device)
        #labels = torch.autograd.Variable(labels).to(device)
        loss = criterion(model(images), labels) 
        running_loss+=loss.item()*labels.shape[0]
        examples+=labels.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
    auc, accuracy, kap = evaluate_model(testset, model, device)
    model_c = copy.deepcopy(model)
    #Find gaidance model   
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate_target)
    for batch_idx, (images, labels) in enumerate(train_loader):
        #model.train()

        images = images.to(device)
        labels = labels.to(device)
        #images = torch.autograd.Variable(images.view(-1,input_size)).to(device)
        #labels = torch.autograd.Variable(labels).to(device)
        loss = criterion(model(images), labels) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
    ###Train base of the local model, keep freeze the head
    del train_loader, images, 
    return model, model_c, auc, accuracy, kap

################################################################################################################################
##Split target dataset 2%
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
        
        # Define augmentation transformations for source domains
        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
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
        ###env = [amazon : , caltech10, dslr, webcam]
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
       
#############################################################################################################################
    cnn.to(args.device_server)
    sources_per_models=dict()                                            
    for i in range(len(sub_sources)):
        sources_per_models[i] = copy.deepcopy(cnn).to(args.device_server)
    target_per_model = dict()
    target_per_model[0] = copy.deepcopy(cnn).to(args.device_server)
    target_accuracy=[]   
      
        
    print("fed traning ......")
    total_time=0

    for epoch in range(args.global_epoch):
        start_time = time.time()
        clients_models = dict()
        gaidance_models = dict()
        for i in range(len(sub_sources)): 
            mod, gmod = local_update_sources(i, args.device_local, sub_sources[i], sources_per_models)

            clients_models[i] = flat_grad(get_param(mod)).to(args.device_server)
            gaidance_models[i] = flat_grad(get_param(gmod)).to(args.device_server)
        del mod, gmod
        mod, gmod, auc, acc, ka = local_update_target(args.device_local, target_train, target_test, target_per_model)
        clients_models[len(sub_sources)] = flat_grad(get_param(mod)).to(args.device_server)
        gaidance_models[len(sub_sources)] = flat_grad(get_param(gmod)).to(args.device_server)
        target_accuracy.append(acc)
        
        for idx1 in range(len(sub_sources)+1):
            sq_inv_res_gmodel=0
            for idx2 in range(len(sub_sources)+1):
                sq_inv_res_gmodel+=inverse_square_euclidean_distance(gaidance_models[idx1], clients_models[idx2])
            
            perW_i = dict()
            for idx2 in range(len(sub_sources)+1):
                perW_i[idx2] = inverse_square_euclidean_distance(gaidance_models[idx1], clients_models[idx2])/sq_inv_res_gmodel
            perMod_i = 0
            for idx3 in range(len(sub_sources)+1):
                perMod_i+=perW_i[idx3]*clients_models[idx3]
                
            ### We have personalized model in 1D tensor format
            perMod_i=makeweights_list(cnn, perMod_i)        
            for index, param in enumerate(cnn.parameters()):
                param.data=perMod_i[index].clone() 
            if idx1 == len(sub_sources):
                target_per_model[idx1] = cnn
            else:
                sources_per_models[idx1] = cnn
                
        del perMod_i, perW_i, clients_models, gaidance_models            

        round_time = time.time() - start_time
        total_time = total_time + round_time
        print("{} ".format(epoch+1), "{:.4f}".format(auc), "{:.4f}".format(acc), "{:.4f}".format(ka), "{:.3f}".format(total_time))
        
    print("Best accuracies in all the target clients:", max(target_accuracy))
            
