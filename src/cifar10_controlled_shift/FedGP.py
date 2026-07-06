import random
import os
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
from torchvision.transforms.functional import rotate
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from collections import defaultdict
from collections import defaultdict
import fedlab
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser() 

parser.add_argument('--Algo', type=str, default='FedGP', help="name \of dataset") #FedDA FedGP
parser.add_argument('--dataset', type=str, default='CIFAR10', help="name \of dataset") 

parser.add_argument('--degree_scarcity', type=float, default=0.5, help="name \of dataset")
parser.add_argument('--noise', type=float, default= 0.9, help="name \of dataset")
parser.add_argument('--dir_alpha', type=float, default= 1 , help="name \of dataset")
parser.add_argument('--num_users', type=int, default=10, help="number of perticipating clients")

parser.add_argument('--batch_sizeS', type=int, default=64, help='batch size') #Fixed
parser.add_argument('--batch_sizeT', type=int, default=16, help='batch size') #Fixed
parser.add_argument('--global_epoch', type=int, default=50, help='local epoch') #Fixednum_epochs
parser.add_argument('--seed', type=int, default=50, help='random seed') ##Fixed
parser.add_argument('--local_epoch', type=int, default=1, help='local epoch') #Fixednum_epochs

 
parser.add_argument('--learning_rate', type=float, default=0.001 , help='choose learning rate for optimizer') 
parser.add_argument('--learning_rate_target', type=float, default=0.0001 , help='choose learning rate for optimizer')
parser.add_argument('--proj_w', type=float, default=0.5 , help='choose learning rate for optimizer')

parser.add_argument('--device_server', type=int, default=4, help='gpu number for server') 
parser.add_argument('--device_local', type=int, default=5, help='gpu number for server')


args = parser.parse_args()
np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False
torch.manual_seed(args.seed) 
random.seed(args.seed)
##########################################################################################################################
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

###########################################################################################################################
class empirical_metrics_batch:
    def __init__(self, target_batch_size, source_grads, target_grads):
        self.target_batch_size = target_batch_size # we don't need this for the source client

        self.source_grads = source_grads # the source grad is simply the average of all batches size list of [(M, )]
        self.target_grads = target_grads # A tensor of size (N, M) where N is the number of batches and M is the dim
        self.target_grad = torch.mean(target_grads, dim=0) # the average of self.target_grads, size of (M, )

        # call self.compute_quantities() to compute the following quantities after getting the above three quantities
        self.target_var = None
        self.source_target_var = [] # lenght of number of source clients
        # self.taus = [] # length of number of source clients
        self.projected_grads_norm_square = []
        self.deltas = []
        
        self.compute_quantities()

    def compute_quantities(self):
        num_batches, dim = self.target_grads.shape
        # compute target variance
        sample_target_var = torch.sum((self.target_grads - self.target_grad) ** 2) / (num_batches - 1) / dim
        self.target_var = sample_target_var / num_batches
        # compute norm of the target gradients
        self.target_norm_square = torch.norm(self.target_grad).item() ** 2 / dim
        # compute source target difference
        for source_grad in self.source_grads:
            sample_source_target_var = torch.sum((self.target_grads - source_grad) ** 2) / num_batches / dim
            self.source_target_var.append(max(sample_source_target_var - sample_target_var, 0.))
            # compute tau
            # eps = 0.0001  # room to numerical error
            # diff = torch.norm(self.target_grad - source_grad)
            # cos_rho = (source_grad * self.target_grad).sum() / torch.norm(self.target_grad) / torch.norm(source_grad)
            # sin_rho = (1 - cos_rho ** 2) ** 0.5
            # print(sin_rho)
            # if diff < eps:
            #     tau = 0
            # else:
            #     tau = (torch.norm(self.target_grad) * sin_rho / diff).item()
            projected_grads = self.target_grads - (torch.sum(self.target_grads * source_grad, dim=1) * source_grad.view([-1, 1])).T / torch.norm(source_grad) ** 2
            projected_grad = self.target_grad - torch.sum(self.target_grad * source_grad) * source_grad / torch.norm(source_grad) ** 2
            projected_grads_var = torch.sum((projected_grads - projected_grad) ** 2) / (num_batches - 1) / dim
            projected_grads_norm_var = torch.mean(torch.norm(projected_grads, dim=1) ** 2) / dim
            self.projected_grads_norm_square.append(max(projected_grads_norm_var - projected_grads_var, 0.))

                
            # compute delta
            inner_products = torch.sum(self.target_grads * source_grad, dim=1)
            delta = torch.sum(inner_products > 0) / num_batches
            self.deltas.append(1 - (1 - delta.item()) / num_batches)
            # self.taus.append(tau)
        
        # print(self.deltas, self.taus, self.source_target_var, self.target_var)
    
    def return_fedda_beta(self):
        return [self.target_var / (self.target_var + s_t_var) for s_t_var in self.source_target_var]
    
    def return_fedgp_with_thresh_beta(self):
        return [self.target_var / (self.target_var + self.deltas[idx] * self.projected_grads_norm_square[idx] + (1-self.deltas[idx]) * self.target_norm_square) for idx in range(len(self.source_grads))]
    
    def return_fedgp_beta(self):
        return [self.target_var / (self.target_var + self.projected_grads_norm_square[idx]) for idx in range(len(self.source_grads))]
        
#######################################################################################################################################
# get the grad updates
def get_model_updates(init_model, new_model):
    init = get_param_list(init_model)
    new = get_param_list(new_model)
    return (new - init)
    
def get_param_list(model):
    m_dict = model.state_dict()
    param = []
    for key in m_dict.keys():
        if m_dict[key].shape != torch.Size([]):
            param.append(m_dict[key].detach().clone().flatten())
    return torch.cat(param)
########################################################################################################################################
def train_source(device, cnn, train_set):
    num_epochs = args.local_epoch
    model=copy.deepcopy(cnn)
    #print("model size:", get_param_list(model).shape)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_dl=torch.utils.data.DataLoader(train_set, batch_size=args.batch_sizeS, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate) #SGD Adam
    grads_all_epochs = []
    model.train()
    for epoch in range(num_epochs):
        num_batches = 0
        model_init = copy.deepcopy(model)
        for batch_idx, (imgs, labels) in enumerate(train_dl):
            num_batches += 1
            optimizer.zero_grad()
            loss = criterion(model(imgs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            
        gf = get_model_updates(model_init, model).detach().cpu()/ num_batches
        gf.to(args.device_server)
        grads_all_epochs.append(gf)
    grads_all_epochs = torch.mean(torch.stack(grads_all_epochs),dim=0)
    #print("source grad:", grads_all_epochs.shape)
    del gf, model_init
    return model, grads_all_epochs

def train_target(device, cnn, train_set):
    num_epochs = args.local_epoch
    model=copy.deepcopy(cnn)
    #print("model size target:", get_param_list(model).shape)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_dl=torch.utils.data.DataLoader(train_set, batch_size=args.batch_sizeT, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate_target) #SGD Adam
    grads_all_epochs = []
    model.train()
    for epoch in range(num_epochs):
        grads = [] # length N - number of batches
        for batch_idx, (imgs, labels) in enumerate(train_dl):
            model_init = copy.deepcopy(model)
            optimizer.zero_grad()
            loss = criterion(model(imgs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            cur_grad = get_model_updates(model_init, model).cpu()
            #print("target grad shape:", cur_grad.shape)
            cur_grad.to(args.device_server)
            grads.append(cur_grad)

        grads = torch.stack(grads) # [Number of batches, m]
        grads_all_epochs.append(grads)
    #print("model size target after training:", get_param_list(model).shape)    
    grads_all_epochs = torch.mean(torch.stack(grads_all_epochs),dim=0)
    grads_all_epochs = grads_all_epochs * (args.learning_rate_target/args.learning_rate)
    #print("target grad:", grads_all_epochs.shape)
    return model, grads_all_epochs
    
def average_weights(w, alpha):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key]).float()
        for i in range(len(w)):
            w_avg[key] += w[i][key] * alpha[i]
    return w_avg

def update_dict(old_model_dict, new_model_dict, alpha):
    new_w = copy.deepcopy(old_model_dict)
    for key in new_w.keys():
        new_w[key] = torch.zeros_like(new_w[key]).float()
        new_w[key] = old_model_dict[key] * alpha + new_model_dict[key] * (1-alpha)
    return new_w

def update_global(n_target_samples, local_models_dict, old_global_model_dict, finetune_global_model_dict, clients_size, clients_size_frac, cur_epoch, beta_GP):
    ret_dict = copy.deepcopy(old_global_model_dict)
    b = beta_GP
    cos = torch.nn.CosineSimilarity()
    for key in ret_dict.keys():
        if ret_dict[key].shape != torch.Size([]):
            global_grad = finetune_global_model_dict[key] - old_global_model_dict[key]
            for idx, local_dict in enumerate(local_models_dict):
                local_grad = local_dict[key] - old_global_model_dict[key]
                cur_sim = cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1))
                if cur_sim > 0:
                    ret_dict[key] = ret_dict[key] + beta_GP[idx] * (args.learning_rate_target/args.learning_rate) * ((n_target_samples/args.batch_sizeT)/(clients_size[idx]/args.batch_sizeS)) * clients_size_frac[idx] * cur_sim * local_grad
                ret_dict[key] = ret_dict[key] + (1-beta_GP[idx]) * global_grad * clients_size_frac[idx]
        else:
            ret_dict[key] = torch.zeros_like(old_global_model_dict[key]).float()
            for idx, local_dict in enumerate(local_models_dict):
                ret_dict[key] += clients_size_frac[idx] * local_dict[key]
    return ret_dict

def update_global_convex(local_models_dict, old_global_model_dict, finetune_global_model_dict, clients_size, clients_size_frac, cur_epoch, beta_DA):
    ret_dict = copy.deepcopy(old_global_model_dict)
    for key in ret_dict.keys():
        if ret_dict[key].shape != torch.Size([]):
            global_grad = finetune_global_model_dict[key] - old_global_model_dict[key]
            for idx, local_dict in enumerate(local_models_dict):
                local_grad = local_dict[key] - old_global_model_dict[key]
                ret_dict[key] = ret_dict[key] + beta_DA[idx] * clients_size_frac[idx] * local_grad
                ret_dict[key] = ret_dict[key] + (1-beta_DA[idx]) * global_grad * clients_size_frac[idx]
        else:
            ret_dict[key] = torch.zeros_like(old_global_model_dict[key]).float()
            for idx, local_dict in enumerate(local_models_dict):
                ret_dict[key] += clients_size_frac[idx] * local_dict[key]
    return ret_dict

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
#################################################################################################################################

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

       
###################################################################################################################################

    print("Fed Training")
    
    clients_size = [len(sub_sources[i]) for i in range(len(sub_sources))]
    clients_size_frac = np.array(clients_size) / sum(clients_size)
    cnn.to(args.device_server)
    total_time=0;    
    # do fedavg for 2 epochs, to have a good initialization
    for ep in range(2):
        models_list = []
        for idx in range(len(sub_sources)):
            mod_s, _ = train_source(args.device_local, cnn, sub_sources[idx])
            models_list.append(mod_s.to(args.device_server))
        models_list = average_weights([model.state_dict() for model in models_list], clients_size_frac)
        cnn.load_state_dict(models_list)
    del models_list, mod_s
    target_accuracy=[]  
    total_time=0;
    for epoch in range(args.global_epoch):
        start_time = time.time()
        local_models = []
        source_grads = []
        target_grads = None
        for idx in range(len(sub_sources)):
            s_mod, source_grad = train_source(args.device_local, cnn, sub_sources[idx])
            s_mod.to(args.device_server)
            source_grads.append(source_grad)
            local_models.append(s_mod)
        del s_mod, source_grad


        if args.Algo == "FedGP":
            t_mod, target_grads = train_target(args.device_local, cnn, target_train)
            t_mod.to(args.device_server)
            auc, acc, ka = evaluate_model(target_test, t_mod, args.device_server)
            target_accuracy.append(acc)

            if args.proj_w > 0:
                metrics = empirical_metrics_batch(args.batch_sizeT, source_grads, target_grads)
                beta_GP = metrics.return_fedgp_beta() 
                #print(beta_GP)
                global_model_dict = update_global(len(target_train), [model.state_dict() for model in local_models], cnn.state_dict(), t_mod.state_dict(), clients_size, clients_size_frac, epoch, beta_GP)
          
                cnn.load_state_dict(global_model_dict)
                del global_model_dict
            else:
                cnn = copy.deepcopy(t_mod)
           
        elif args.Algo == "FedDA":
            t_mod, target_grads = train_target(args.device_local, cnn, target_train)
            t_mod.to(args.device_server)
            auc, acc, ka = evaluate_model(target_test, t_mod, args.device_server)
            target_accuracy.append(acc)

            if args.proj_w > 0:
                metrics = empirical_metrics_batch(args.batch_sizeT, source_grads, target_grads)
                beta_DA = metrics.return_fedda_beta()
                #print(beta_DA)
                global_model_dict = update_global_convex([model.state_dict() for model in local_models], cnn.state_dict(), t_mod.state_dict(), clients_size, clients_size_frac, epoch, beta_DA)
                cnn.load_state_dict(global_model_dict)
            else:
                cnn = copy.deepcopy(new_model)
        #auc, acc, ka = evaluate_model(target_test, cnn, args.device_server)
        #target_accuracy.append(acc)
        round_time = time.time() - start_time
        total_time = total_time + round_time
        print("{} ".format(epoch+1), "{:.4f}".format(auc), "{:.4f}".format(acc), "{:.4f}".format(ka), "{:.3f}".format(total_time))
    print("Best accuracies in all the target clients:", max(target_accuracy))        
 
