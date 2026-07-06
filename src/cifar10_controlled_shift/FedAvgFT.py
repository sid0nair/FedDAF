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
import fedlab
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile
import torchvision.datasets.folder
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser() 
parser.add_argument('--dataset', type=str, default='CIFAR10', help="name \of dataset") 

parser.add_argument('--degree_scarcity', type=float, default=0.05, help="name \of dataset")
parser.add_argument('--noise', type=float, default= 0.6, help="name \of dataset")
parser.add_argument('--dir_alpha', type=float, default= 1 , help="name \of dataset")
parser.add_argument('--num_users', type=int, default=10, help="number of perticipating clients") 


parser.add_argument('--batch_sizeS', type=int, default=64, help='batch size') #Fixed
parser.add_argument('--batch_sizeT', type=int, default=16, help='batch size') #Fixed
parser.add_argument('--global_epoch', type=int, default=50, help='local epoch') #Fixed
parser.add_argument('--seed', type=int, default=50, help='random seed') ##Fixed

parser.add_argument('--learning_rate', type=float, default= 0.01 , help='choose learning rate for optimizer')
parser.add_argument('--learning_rate_target', type=float, default=0.001 , help='choose learning rate for optimizer')

parser.add_argument('--device_server', type=int, default=2, help='gpu number for server')
parser.add_argument('--device_local', type=int, default=1, help='gpu number for server')

args = parser.parse_args()
np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False
torch.manual_seed(args.seed) 
random.seed(args.seed)
ImageFile.LOAD_TRUNCATED_IMAGES = True
###########################################################################################
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out   

######################################################################################
def split_cifar100(test_dataset):

    # Extract the labels
    labels = np.array(test_dataset.targets)
    
    # Get the stratified train-test split indices
    train_indices, val_indices = train_test_split(
        np.arange(len(labels)), 
        test_size=0.8, 
        stratify=labels,  # Ensure proportional class distribution
        random_state=args.seed
    )
    
    # Create the subsets
    subset_1 = torch.utils.data.Subset(test_dataset, train_indices)
    subset_2 = torch.utils.data.Subset(test_dataset, val_indices)   
    return subset_1, subset_2
    
def RandomSplit(D, ratio):
    D_size = len(D)
    train_indices, _ = train_test_split(
        np.arange(D_size), 
        test_size= (1 - ratio),  # 0.25 of subset_1 will be split into subset_1b, so subset_1a will have 60% of the original dataset
        random_state=args.seed  # For reproducibility
    )
    return torch.utils.data.Subset(D, train_indices)
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise 

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
if __name__ == '__main__':
    print("preparing data")
    if args.dataset == 'CIFAR10':  
        img_size=32
        n_channel=3
        n_classes=100
        input_size=32*32*3
        train_augmentations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_augmentations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), AddGaussianNoise(mean=0.0, std=args.noise) ])        

        trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform= train_augmentations)
        
        testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=test_augmentations)
        target_train, target_test = split_cifar100(testset)
        del testset
        target_train = RandomSplit(target_train, args.degree_scarcity)
        
        hetero_dir_part = CIFAR10Partitioner(trainset.targets,
                                     args.num_users,
                                     balance=None,
                                     partition="dirichlet",
                                     dir_alpha=args.dir_alpha,
                                     seed=args.seed)
        
        cm=hetero_dir_part.client_dict
        sub_sources = []
        for i in range(args.num_users):
            data_inds = cm[i]
            sub_sources.append(torch.utils.data.Subset(trainset, data_inds))
        del trainset, cm
        cnn = ResNet9()
        #cnn = models.resnet18(pretrained=False)
        #num_ftrs = cnn.fc.in_features
        #cnn.fc = torch.nn.Linear(num_ftrs, 10) 
    
#############################################################################################
    print("Fed Training")
    cnn.to(args.device_server)
    target_accuracy=[]   
    total_time=0;
    for epoch in range(args.global_epoch):
        start_time = time.time()
        ModTarget = local_update(args.device_local, cnn, target_train, args.learning_rate_target, args.batch_sizeT)
        ModTarget = ModTarget.to(args.device_server)
        auc, acc, ka = evaluate_model(target_test, ModTarget, args.device_server)
        ModTarget = flat_grad(get_param(ModTarget))
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
        target_accuracy.append(acc)
        round_time = time.time() - start_time
        total_time = total_time + round_time
        print("{} ".format(epoch+1), "{:.4f}".format(auc), "{:.4f}".format(acc), "{:.4f}".format(ka), "{:.3f}".format(total_time))
        
    print("Best accuracies in all the target clients:", max(target_accuracy))
