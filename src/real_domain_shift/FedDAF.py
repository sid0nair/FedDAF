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

parser.add_argument('--learning_rate', type=float, default= 0.01 , help='choose learning rate for optimizer')
parser.add_argument('--learning_rate_target', type=float, default= 0.001 , help='choose learning rate for optimizer')
parser.add_argument('--k', type=float, default=5 , help='Gompertz function parameter (mu in the paper)')

parser.add_argument('--mu', type=float, default=0.001, help='proximal regularization coefficient (lambda in the paper)') ##Fixed
#parser.add_argument('--num_per_class', type=int, default=500, help= 'choose alpha') ##Fixed

parser.add_argument('--device_server', type=int, default=0, help='gpu number for server')
parser.add_argument('--device_local', type=int, default=1, help='gpu number for server')

args = parser.parse_args()
np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False
torch.manual_seed(args.seed) 
random.seed(args.seed)
ImageFile.LOAD_TRUNCATED_IMAGES = True
###########################################################################################
#For Couour-MNIST
def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, ['+80%', '+90%', '-90%'],
                                         self.color_dataset, (2, 28, 28,), 10)  # 10 classes for MNIST

        self.input_shape = (2, 28, 28,)
        self.num_classes = 10  # Update for multiclass MNIST

    def get_color_shift_probability(self, environment):
        """Return the probability of a color shift depending on the environment."""
        if environment == '+90%':
            return 0.9  # High probability of color shift in the target environment
        elif environment == '+80%':
            return 0.8  # Moderate probability in one of the source environments
        elif environment == '-90%':
            return 0.1  # Low probability in the other source environment
        else:
            raise ValueError(f"Unknown environment: {environment}")

    def color_dataset(self, images, labels, environment):
        # Normalize the labels into a float format (for binary classification, `0 or 1` would work; here, we're using MNIST)
        labels = (labels.float())  # Use original labels as they are (0-9 for MNIST)
        
        # Flip label with probability 0.25 (this could be customized)
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Get the color shift probability based on the environment
        shift_probability = self.get_color_shift_probability(environment)

        # Apply color shift based on the label
        colors = self.torch_xor_(labels, self.torch_bernoulli_(shift_probability, len(labels)))

        # Ensure colors contains only 0 or 1 (binary values)
        colors = colors.clamp(min=0, max=1).long()  # Make sure the values are either 0 or 1

        # Sanity check: Ensure that colors only contain 0 or 1
        assert torch.all((colors == 0) | (colors == 1)), "Colors tensor must only contain 0 or 1 values."

        # Stack images into two channels to simulate a color shift
        images = torch.stack([images, images], dim=1)  # Shape [N, 2, 28, 28]

        # Apply the color shift by zeroing out one of the color channels
        # (1 - colors).long() gives us 0 or 1 values to select between the two channels
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)  # Normalize the image to [0, 1]
        y = labels.view(-1).long()  # Ensure labels are in long format for PyTorch

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

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
                
        print(len(target_data), len(sub_sources[0]), len(sub_sources[1]), len(sub_sources[2]))
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
        ###env = [amazon, caltech10, dslr, webcam] 958 1123 157 295 ## first two 0.2 and last two 0.02
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
        Target_index = 0
        target_data = data_list[Target_index]
        sub_sources = [data_list[i] for i in p if i != Target_index]
        del data_list  
        
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
        cnn.fc = torch.nn.Linear(num_ftrs, 10)
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
