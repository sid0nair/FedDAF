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
from torchvision import transforms
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
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()  

parser.add_argument('--dataset', type=str, default='CIFAR10', help="name \of dataset") 

parser.add_argument('--degree_scarcity', type=float, default=0.25, help="name \of dataset")
parser.add_argument('--noise', type=float, default= 0.9, help="name \of dataset")
parser.add_argument('--dir_alpha', type=float, default= 1 , help="name \of dataset")
parser.add_argument('--num_users', type=int, default=10, help="number of perticipating clients") 

parser.add_argument('--batch_sizeS', type=int, default=64, help='batch size') #Fixed
parser.add_argument('--batch_sizeT', type=int, default=16, help='batch size') #Fixed
parser.add_argument('--global_epoch', type=int, default=50, help='local epoch') #Fixed
parser.add_argument('--seed', type=int, default=50, help='random seed') ##Fixed

parser.add_argument('--learning_rate', type=float, default= 0.01 , help='choose learning rate for optimizer')
parser.add_argument('--learning_rate_target', type=float, default= 0.001 , help='choose learning rate for optimizer')
parser.add_argument('--k', type=float, default= 5 , help='Gompertz function parameter (mu in the paper)')

parser.add_argument('--mu', type=float, default=0.001, help='proximal regularization coefficient (lambda in the paper)') ##Fixed
#parser.add_argument('--num_per_class', type=int, default=500, help= 'choose alpha') ##Fixed

parser.add_argument('--device_server', type=int, default=5, help='gpu number for server')
parser.add_argument('--device_local', type=int, default=4, help='gpu number for server')

args = parser.parse_args()
np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False
torch.manual_seed(args.seed) 
random.seed(args.seed)
######################################################################################
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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

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
######################################################################################
def GenerateSamples(data_set, num_comp, num_per_class, num_channel, img_size):  
    data_loader=torch.utils.data.DataLoader(data_set, batch_size=len(data_set), shuffle=True, num_workers=2)
    x_train = next(iter(data_loader))[0].numpy() 
    y_train = next(iter(data_loader))[1].numpy()
    x_train = np.reshape(x_train, [x_train.shape[0], -1])
    y_train = np.reshape(y_train, [y_train.shape[0], ])
    classes = set(y_train)
    GMMs = []
    x_gen = []
    y_gen = []
    for c in classes:
        x = x_train[y_train == c]
        gmm = GaussianMixture(n_components=num_comp, covariance_type='diag', max_iter=150,
                              verbose=0).fit(x[np.random.permutation(x.shape[0])[:2000]])
        x_gen.append(gmm.sample(num_per_class)[0].reshape(num_per_class, num_channel, img_size, img_size))  
        y_gen.append(np.ones(num_per_class) * c)
        
    gset = torch.utils.data.TensorDataset(torch.tensor(np.vstack(x_gen), dtype=torch.float32), torch.tensor(np.hstack(y_gen), dtype=torch.uint8)) 
    del x_gen, y_gen, data_loader
    return gset
######################################################################################
def local_update_source(device, cnn, train_set):
    model=copy.deepcopy(cnn)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_loader=torch.utils.data.DataLoader(train_set, batch_size=args.batch_sizeS, shuffle=True, num_workers=2)
    optimiz1=torch.optim.SGD(params=model.parameters(), lr=args.learning_rate)
    for batch_idx, (images, labels) in enumerate(train_loader):
        model.train()
        images = images.to(device)
        labels = labels.to(device)
        loss = criterion(model(images), labels) 
        optimiz1.zero_grad()
        loss.backward()
        optimiz1.step()
    del train_loader, images
    return model

def FindPersonalizedModel(model_l, globalmod, dataset, device):
    criterion = nn.CrossEntropyLoss()
    train_loader=torch.utils.data.DataLoader(dataset, batch_size=args.batch_sizeT, shuffle=True, num_workers=2) 
    cos = torch.nn.CosineSimilarity(dim=0)    
    model_g = copy.deepcopy(globalmod).to(device)
    grad_glob = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        loss = criterion(model_g(images), labels) 
        model_g.zero_grad()
        loss.backward()
        g= flat_grad([para.grad for para in model_g.parameters()])    
        grad_glob+= g.detach().clone()
    grad_glob=grad_glob/len(train_loader)


    grad_local = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        loss = criterion(model_l(images), labels) 
        model_l.zero_grad()
        loss.backward()
        g= flat_grad([para.grad for para in model_l.parameters()])    
        grad_local+= g.detach().clone()
    grad_local=grad_local/len(train_loader)

    sim=cos(grad_local, grad_glob)
    a = torch.acos(sim)
    #Gompertz function
    sim = (1 - (torch.exp(-torch.exp(-args.k*(a-1)))))
    del train_loader, images, g, grad_local, grad_glob
    mod = sim*flat_grad(get_param(model_g)) + (1-sim)*flat_grad(get_param(model_l))
    del model_g
    mod = makeweights_list(model_l, mod)        
    for index, param in enumerate(model_l.parameters()):
        param.data=mod[index].clone() 
    del mod


def local_update_target(device, trainset, testset, model_clients, global_model):

    if model_clients[0] is not None:
        model = model_clients[0]
        FindPersonalizedModel(model, global_model, trainset, device)
    else:
        model = copy.deepcopy(cnn).to(device)
    #auc, accuracy, kap = evaluate_model(testset, model, device)
    criterion = nn.CrossEntropyLoss()
    model_0=copy.deepcopy(global_model)
    model_0.to(device)
    train_loader=torch.utils.data.DataLoader(trainset, batch_size=args.batch_sizeT, shuffle=True, num_workers=2) ####genset
    optimiz=torch.optim.SGD(params=model.parameters(), lr=args.learning_rate_target)
    for batch_idx, (images, labels) in enumerate(train_loader):
        model.train()
        images = images.to(device)
        labels = labels.to(device)
        loss = criterion(model(images), labels) 
        loss+=(args.mu/2)*difference_models_norm_2(model,model_0)
        optimiz.zero_grad()
        loss.backward()
        optimiz.step()

    auc, accuracy, kap = evaluate_model(testset, model, device)
    model_clients[0] = model
    del train_loader, images, model, model_0
    return model_clients[0], auc, accuracy, kap
    
def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters
    """
    
    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())
    
    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) 
        for i in range(len(tensor_1))])
    
    return norm
    
def compute_distance(tensor1, tensor2):
    # Compute Euclidean distance between tensor1 and tensor2
    return torch.norm(tensor1 - tensor2)

def aggregate_with_softmax(tensor_list, reference_tensor):
    # Step 1: Compute the distances from each tensor in the list to the reference tensor
    distances = [compute_distance(tensor, reference_tensor) for tensor in tensor_list]
    distances = torch.tensor(distances)  # Convert the list of distances to a tensor
    
    # Step 2: Normalize distances using Softmax. We negate the distances to prioritize smaller distances.
    softmax_weights = F.softmax(-distances, dim=0)  # Negative sign gives higher weights to closer tensors
    
    # Step 3: Aggregate tensors based on the softmax weights
    aggregated_tensor = sum(w * t for w, t in zip(softmax_weights, tensor_list))
    
    return aggregated_tensor
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
    model_target = dict()
    model_target[0] = None
    
    target_accuracy=[]
    total_time=0;
    
    for epoch in range(args.global_epoch):
        start_time = time.time()
        #Perform local training for target client
        l_mod, auc, acc, ka =local_update_target(args.device_local, target_train, target_test, model_target, cnn)
        l_mod = flat_grad(get_param(l_mod)).to(args.device_server)
        target_accuracy.append(acc)
        source_models=[]        
        for i in range(len(sub_sources)): 
            mod = local_update_source(args.device_local, cnn, sub_sources[i])
            mod = mod.to(args.device_server)
            mod = flat_grad(get_param(mod))
            source_models.append(mod)
            
        average = aggregate_with_softmax(source_models, l_mod)
        del source_models, l_mod, mod
        #### Update the global model of sources based on target model
        average = makeweights_list(cnn, average)        
        for index, param in enumerate(cnn.parameters()):
            param.data=average[index].clone() 
        del average
        round_time = time.time() - start_time
        total_time = total_time + round_time
        print("{} ".format(epoch+1), "{:.4f}".format(auc), "{:.4f}".format(acc), "{:.4f}".format(ka), "{:.3f}".format(total_time))
        
    print("Best accuracies in all the target clients:", max(target_accuracy))
