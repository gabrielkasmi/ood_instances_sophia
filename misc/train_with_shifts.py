# -*- coding: utf-8 -*-

"""
Trains a model. Computes the representation. Evaluates the model OOD

"""

from re import I
import torch
import numpy as np
import PIL
import random
import os
from src import dataset, helpers, custom_dataloaders
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import Inception3
from torchvision import transforms
import torchvision
from tqdm import tqdm
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml 
import argparse
import json

# Ignore warining
import warnings
warnings.filterwarnings("ignore")


# output dictionnary

output = {} 


# Companion functions

def return_f1(precision, recall):

    return 2 * (np.array(precision) * np.array(recall)) / (np.array(precision) + np.array(recall))

def plot_curves(precision, recall):
    """
    add points to compute the p/r curves
    """
    
    precisions_plot = [[0]]
    recall_plot = [[recall[0]]]

    precisions_plot.append(precision)
    recall_plot.append(recall)



    precisions_plot.append([precision[-1]])
    recall_plot.append([0.5])

    precisions_plot = sum(precisions_plot, [])
    recall_plot = sum(recall_plot, [])
    
    return precisions_plot, recall_plot


# Setting up the seed
seed = 42

torch.backends.cudnn.deterministic = True
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Load the configuration file
config = 'config.yml'

with open(config, 'rb') as f:
    configuration = yaml.load(f, Loader=yaml.FullLoader)

# Retrieve the directories
models_dir = configuration.get("models_dir") 
files_dir = configuration.get("files_dir") 
figs_dir = configuration.get("figs_dir") 
data_dir = configuration.get("dataset_dir") 

# Retrieve the options for the model
pretrained : configuration.get('pretrained', True)
model_family = configuration.get('model', 'inception')
model_weight = configuration.get('model_weight')
image_size = configuration.get('image_size')
# label_name = configuration.get('label_name', 'labels.csv')

output_directory = configuration.get("output_directory", 'results') 


# If the directory does not exist, create the temporary folder 
# where the checkpoints will be stored.

weights_dir = 'weights'
checkpoints_dir = os.path.join(weights_dir, 'losses')

if not os.path.isdir(weights_dir):
    os.mkdir(weights_dir)

if not os.path.isdir(checkpoints_dir):
    os.mkdir(checkpoints_dir)

# Arguments
parser = argparse.ArgumentParser(description = 'Model training')

parser.add_argument('--device', default = 'cuda:0', help = "GPU device")

parser.add_argument('--n_epochs', default = 30, help = "Number of training epochs", type=int)
parser.add_argument('--batch_size', default = 64, help = "Batch size", type=int)
parser.add_argument('--print_every', default = 20, help = "Evaluate the model every print_every iterations.", type=int)
parser.add_argument('--name', default = 'model', help = "Name of the model.")


# Additional arguments with which we will edit the label files
parser.add_argument('--array_type', default = None, help = "Class of arrays to consider.")
parser.add_argument('--max_count', default = 10000, help = 'Maximal number of samples to sample from.', type = int)
parser.add_argument('--source_domain', default = None, help = "The source domain. The target domain will be deduced.")

parser.add_argument('--bootstrap', default = 50, help = "Number of bootstrap iterations to estimate the dimensionality", type = int)
parser.add_argument('--n_iter', default = 10, help = "Number of train/dimensionality estimation/test iterations", type = int)

args = parser.parse_args()

if args.array_type is None:
    print('Please enter a value for the type of array.')
    raise ValueError

if args.source_domain is None:
    print('Please enter a value for the source domain.')
    raise ValueError


# This dictionnary maps the types of arrays with the corresponding label files, in and out of distribution
corresponding_array_labels = {
    'LB' : ['LB_fields_labels.csv', 'LB_forest_labels.csv'],
    'LN' : ['LN_fields_labels.csv', 'LN_forest_labels.csv'],
    'SB' : ['SB_fields_labels.csv', 'SB_forest_labels.csv'],
    'SN' : ['SN_fields_labels.csv', 'SN_forest_labels.csv'],
    'large' : ['field_large_labels.csv', 'forest_large_labels.csv'],
    'small' : ['field_small_labels.csv', 'forest_small_labels.csv'],
    'blue' : ['blue_field_labels.csv', 'blue_forest_labels.csv'],
    'black' : ['black_field_labels.csv', 'black_forest_labels.csv']
}

# Set up the in and out domain label files. 
array_labels = corresponding_array_labels[args.array_type]

in_domain_index = array_labels.index([item for item in array_labels if args.source_domain in item][0]) # these lists return one item
out_domain_index = array_labels.index([item for item in array_labels if args.source_domain not in item][0]) 

# get the name of the files
in_domain_label = array_labels[in_domain_index]
out_domain_label = array_labels[out_domain_index]

# Transforms 
# If the model is resnet, then we crop the image to be 224 * 224
# If the model is custom, then the size should be specified by the user.

if model_family == 'inception': 
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation((90,90)),
        torchvision.transforms.RandomRotation((-90,-90)),
        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
elif model_family == 'resnet':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation((90,90)),
        torchvision.transforms.RandomRotation((-90,-90)),
        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
elif model_family == 'custom':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation((90,90)),
        torchvision.transforms.RandomRotation((-90,-90)),
        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def train():

    # Data

    # Load the data

    # Load the correct image size
    if model_family == 'resnet':
        dataset_dir = os.path.join(data_dir, 'PITW_224')
    elif model_family == 'inception':
        dataset_dir = os.path.join(data_dir, 'PITW_299')

    training_path = os.path.join(dataset_dir, 'train')
    annotations_train = os.path.join(training_path, in_domain_label)
    
    validation_path = os.path.join(dataset_dir, 'validation')
    annotations_validation = os.path.join(validation_path, in_domain_label)

    test_path = os.path.join(dataset_dir, 'test')
    annotations_test = os.path.join(test_path, in_domain_label)

    training_data = dataset.BDPVClassificationDataset(annotations_train, training_path, transform = transforms, max_samples = args.max_count)
    validation_data = dataset.BDPVClassificationDataset(annotations_validation, validation_path, transform = transforms, max_samples = args.max_count)
    test_data = dataset.BDPVClassificationDataset(annotations_test, test_path, transform = transforms, max_samples = args.max_count)

    # Initialize the data

    train = DataLoader(training_data, batch_size = args.batch_size, shuffle = True)
    val = DataLoader(validation_data, batch_size = args.batch_size, shuffle = True)  
    test = DataLoader(test_data, batch_size = args.batch_size, shuffle = True)  

    dataloader = {"train": train, 'val' : val, 'test' : test}

    # Initialize the model

    if model_family == 'inception':

        # Load the model fine-tuned on NRW
        model = Inception3(num_classes = 2, aux_logits = True, transform_input = False, init_weights = True) # Load the architecture
        checkpoint = torch.load(model_weight, map_location = args.device) # Load the weights
        model.load_state_dict(checkpoint['model_state_dict']) # Upload the weights in the model
        model = model.to(args.device) # move the model to the device

    elif model_family == 'resnet':
        # Load the model and send it to the GPU
        model = torchvision.models.resnet50(pretrained = True)

        # Last layer should have an output shape of 2.
        model.fc = nn.Sequential(
               nn.Linear(2048, 2),
               nn.ReLU(inplace=True))
               
        model = model.to(args.device)

    elif model_family == 'custom':
        # Load the model and send it to the GPU
        model = torch.load(model_weight)
        model = model.to(args.device)

    # Train

    # Initialization of the model

    # Layers to update
    for param in model.parameters():
        param.requires_grad = True
        
    # Criterion 
    criterion = nn.BCELoss()

    # Parameters to update and optimizer
    params_to_update = []

    for _, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            
    optimizer = optim.Adam(params_to_update, lr = 0.0001)


    # Training

    running_loss = 0
    waiting = 0
    steps = 0
    early_stop = False
    train_losses, test_losses = [], []

    threshold = 0.5 # Threshold set by default. Will be fine tuned afterwards


    for epoch in range(args.n_epochs):
        for inputs, labels, _ in tqdm(dataloader["train"]):

            steps += 1
            labels = labels.to(torch.float32)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            
            optimizer.zero_grad()
            
            if model_family == 'inception':
                # Accomodate for the particular architecture of inception

                outputs, aux_outputs = model(inputs)
                outputs, aux_outputs = F.softmax(outputs, dim=1), F.softmax(aux_outputs, dim=1)
                outputs, aux_outputs = outputs[:,1], aux_outputs[:,1]

                loss = criterion(outputs, labels) + 0.4 * criterion(aux_outputs, labels)
            
            else: 
                outputs = model(inputs)
                outputs = F.softmax(outputs, dim=1)
                outputs = outputs[:,1]
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


            # Evaluate the model
            if steps % args.print_every == 0:

                test_loss = 0
                model.eval()

                with torch.no_grad():

                    true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0
                    total = 0

                    for inputs, labels, names in dataloader["val"]:

                        labels = labels.to(torch.float32)
                        inputs, labels = inputs.to(args.device), labels.to(args.device)
                        outputs = model.forward(inputs)
                        outputs = F.softmax(outputs, dim=1) # the model returns the unnormalized probs. Softmax it to get probs
                        outputs = outputs[:,1]
                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item()
                        predicted = (outputs >= threshold).long()  # return 0 or 1 if an array has been detected

                        # compute the accuracy of the classification

                        tp, fp, tn, fn = helpers.confusion(predicted, labels)
                        true_positives += tp
                        false_positives += fp
                        true_negatives += tn
                        false_negatives += fn
                        
                train_total = 100 * running_loss / len(dataloader["train"])
                test_total = 100 * test_loss / len(dataloader["val"])
                
                train_losses.append(train_total)
                test_losses.append(test_total)
                
                # Add to the SummaryWriter            
                # loss_writer.add_scalar("train", train_total, steps)
                # loss_writer.add_scalar("test", test_total, steps)
                
                # Compute the F1 score
                precision = np.divide(true_positives, (true_positives + false_positives))
                recall = np.divide(true_positives, (true_positives + false_negatives))
                
                f1 = 2 * np.divide((precision * recall), (precision + recall))
                
                print(
                    f"Epoch {epoch+1}/{args.n_epochs}..."
                    f"Step {steps}...."
                    f"Train loss : {train_total:.3f}......"
                    f"Val loss : {test_total:.3f}......"
                    f"F1 score : {f1:.3f}"               
                )
                running_loss = 0
                model.train()
                
                # early stopping condition
                # if the model fails to improve for five subsequent epochs on the
                # validation set, we stop the training

                if steps == args.print_every: # first time on the validation set
                    min_val_loss = test_total
                    best_model = args.name + '_' + str(steps) + '.pth'

                    # Save the model
                    torch.save(model, os.path.join(checkpoints_dir, best_model))
                else:

                    if not test_total < min_val_loss:
                        waiting += 1

                    else: # save the model, erase the former best model
                        best_model = args.name + '_' + str(steps) + '.pth'
                        torch.save(model, os.path.join(checkpoints_dir, best_model))
                        waiting = 0 # Reset the number of epochs we have to wait.
                        min_val_loss = test_total # Uptdate the new minimum loss
                
                if waiting == 8:
                    early_stop = True
                    print('Model failed to improve for 5 subsequent epochs on the validation dataset.')
                    break

        if early_stop : # early stop if necessary.
            print('Training interrupted.')
            model.eval()
            break


    # Save the best checkpoint as the best model

    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)
    
    # Load the best model 
    model = torch.load(os.path.join(checkpoints_dir, best_model))
    model = model.to(args.device)

    # Save it
    best_model_name = args.name + '.pth'
    torch.save(model, os.path.join(models_dir, best_model_name))

    # Fine tune the classification threshold
    print('Model trained. Now computing the precision and recall curves on the test set.')


    # Thresholds to be considered
    results_models = {}
    thresholds = np.linspace(0.01,.99, 99)

    model.eval()

    # Now we compute the precision/recall curve on the test set directly for 
    # comprison with the out domain labels

    # Forward pass on the validation dataset and accumulate the probabilities 
    # in a single vector.

    probabilities = []
    all_labels = []

    with torch.no_grad():

        for data in tqdm(dataloader["test"]):
            # i +=1
            images, labels, _ = data

            # move the images to the device
            images = images.to(args.device)

            labels = labels.detach().cpu().numpy()
            all_labels.append(list(labels))

            # calculate outputs by running images through the network and computing the prediction using the threshold
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy() # the model returns the unnormalized probs. Softmax it to get probs
            probabilities.append(list(probs[:,1]))

    # Convert the probabilities and labels as an array
    probabilities = sum(probabilities, [])
    probabilities = np.array(probabilities)

    labels = sum(all_labels, [])
    labels = np.array(labels)

    total = len(labels)

    # Save the results
    for threshold in thresholds:

        # new key 
        results_models[threshold] = {}

        true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0

        # treshold the probabilities to get the predictions
        predicted = np.array(probabilities > threshold, dtype = int)

        # compute the true positives, true negatives, false positives and false negatives
        tp, fp, tn, fn = helpers.confusion(predicted, labels)
        true_positives += tp
        false_positives += fp
        true_negatives += tn
        false_negatives += fn

        # store the results
        results_models[threshold]['true_positives'] = true_positives
        results_models[threshold]['false_positives'] = false_positives
        results_models[threshold]['true_negatives'] = true_negatives
        results_models[threshold]['false_negatives'] = false_negatives
        results_models[threshold]['accuracy'] = (true_positives + true_negatives) / total
    
    # Compute the precision and recall
    precision, recall = helpers.compute_pr_curve(results_models)

    # Compute the F1 score
    f1 = return_f1(precision, recall)

    # Determine the best treshold (that maximizes F1) and the corresponding precision and recall

    best_threshold = thresholds[np.argmax(f1)]

    print('Precision and recall curve computed on the test set.')

    # Now fill the dictionnaro of results

    train_outputs = {}

    train_outputs['precision'] = list(precision)
    train_outputs['recall'] = list(recall)
    train_outputs['f1'] = list(f1)
    train_outputs['best_threshold'] = best_threshold

    return train_outputs, best_model

def dimensionality_estimation(best_model):
    """
    computes the dimensionality of the array/background features
    on the test set of training distribution
    """

    # Get the model and remove the last layer (replace it with identity.)

    model = torch.load(os.path.join(checkpoints_dir, best_model))
    model.fc = nn.Identity()
    model = model.to(args.device)

    # set up the data

    # Load the correct image size
    if model_family == 'resnet':
        dataset_dir = os.path.join(data_dir, 'PITW_224/test') # should be path/to/test_set
    elif model_family == 'inception':
        dataset_dir = os.path.join(data_dir, 'PITW_299/test')

     
    array = custom_dataloaders.MutualArrayInformation(dataset_dir, array_type = args.array_type)
    background = custom_dataloaders.MutualBackgroundInformation(dataset_dir, background_type = args.source_domain) # misses a S sometimes.

    data_array = DataLoader(array, batch_size = 256, shuffle = True) # batch size manually set to 512 as it is the maximum admissible by the GPU
    data_background = DataLoader(background, batch_size = 256, shuffle = True)


    # empty list that will contain the dimensionality estimations
    bootstrapped_estimations = []

    for _ in tqdm(range(args.bootstrap)):
        # compute representations
        outputs, factor_list = custom_dataloaders.compute_representations(data_array, data_background, model, args)

        # dimensionality estimation
        dims, _ = custom_dataloaders.dim_est(outputs, factor_list, 3, 2) # Number of factors and residual index are fixed.

        bootstrapped_estimations.append(dims)

    return bootstrapped_estimations

def ood_evaluation(best_model):
    """
    evaluates the model on the target domain. Takes the best model as input.

    """

    # Load the correct image size
    if model_family == 'resnet':
        dataset_dir = os.path.join(data_dir, 'PITW_224')
    elif model_family == 'inception':
        dataset_dir = os.path.join(data_dir, 'PITW_299')

    training_path = os.path.join(dataset_dir, 'train')
    annotations_train = os.path.join(training_path, out_domain_label)
    
    validation_path = os.path.join(dataset_dir, 'validation')
    annotations_validation = os.path.join(validation_path, out_domain_label)

    test_path = os.path.join(dataset_dir, 'test')
    annotations_test = os.path.join(test_path, out_domain_label)

    training_data = dataset.BDPVClassificationDataset(annotations_train, training_path, transform = transforms, max_samples = args.max_count)
    validation_data = dataset.BDPVClassificationDataset(annotations_validation, validation_path, transform = transforms, max_samples = args.max_count)
    test_data = dataset.BDPVClassificationDataset(annotations_test, test_path, transform = transforms, max_samples = args.max_count)

    # Initialize the data

    train = DataLoader(training_data, batch_size = args.batch_size, shuffle = True)
    val = DataLoader(validation_data, batch_size = args.batch_size, shuffle = True)  
    test = DataLoader(test_data, batch_size = args.batch_size, shuffle = True)  

    dataloader = {"train": train, 'val' : val, 'test' : test}

    # load the model

    model = torch.load(os.path.join(checkpoints_dir, best_model))
    model = model.to(args.device)

    # Thresholds to be considered
    results_models = {}
    thresholds = np.linspace(0.01,.99, 99)

    model.eval()

    # Now we compute the precision/recall curve on the test set directly for 
    # comprison with the out domain labels

    # Forward pass on the validation dataset and accumulate the probabilities 
    # in a single vector.

    probabilities = []
    all_labels = []

    with torch.no_grad():

        for data in tqdm(dataloader["test"]):
            # i +=1
            images, labels, _ = data

            # move the images to the device
            images = images.to(args.device)

            labels = labels.detach().cpu().numpy()
            all_labels.append(list(labels))

            # calculate outputs by running images through the network and computing the prediction using the threshold
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy() # the model returns the unnormalized probs. Softmax it to get probs
            probabilities.append(list(probs[:,1]))

    # Convert the probabilities and labels as an array
    probabilities = sum(probabilities, [])
    probabilities = np.array(probabilities)

    labels = sum(all_labels, [])
    labels = np.array(labels)

    total = len(labels)

    # Save the results
    for threshold in thresholds:

        # new key 
        results_models[threshold] = {}

        true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0

        # treshold the probabilities to get the predictions
        predicted = np.array(probabilities > threshold, dtype = int)

        # compute the true positives, true negatives, false positives and false negatives
        tp, fp, tn, fn = helpers.confusion(predicted, labels)
        true_positives += tp
        false_positives += fp
        true_negatives += tn
        false_negatives += fn

        # store the results
        results_models[threshold]['true_positives'] = true_positives
        results_models[threshold]['false_positives'] = false_positives
        results_models[threshold]['true_negatives'] = true_negatives
        results_models[threshold]['false_negatives'] = false_negatives
        results_models[threshold]['accuracy'] = (true_positives + true_negatives) / total
    
    # Compute the precision and recall
    precision, recall = helpers.compute_pr_curve(results_models)

    # Compute the F1 score
    f1 = return_f1(precision, recall)

    # Determine the best treshold (that maximizes F1) and the corresponding precision and recall

    best_threshold = thresholds[np.argmax(f1)]

    ood_outputs = {}

    ood_outputs['precision'] = list(precision)
    ood_outputs['recall'] = list(recall)
    ood_outputs['f1'] = list(f1)
    ood_outputs['best_threshold'] = best_threshold
    
    return ood_outputs


def main():

    print('******** ARRAY TYPE {} AND SOURCE DOMAIN {}********'.format(args.array_type, args.source_domain))

    output = {}

    for i in range(args.n_iter):

        output[str(i)] = {}

        # train the model
        train_outputs, best_model = train()

        output[str(i)]['in_domain'] = train_outputs

        print('Model has been trained.')

        # compute the representations on the training distribution
        # (bootstrap'd)
        dimensions = dimensionality_estimation(best_model)

        output[str(i)]['dimensions'] = dimensions


        print("Dimensionality estimation complete.")

        # evaluate performance on the target domain
        ood_outputs = ood_evaluation(best_model)

        output[str(i)]['ood'] = ood_outputs

        print("OOD performance evaluated.")

        # save the outputs in a dedicated json file

    out_name = 'train_with_shifts_{}_{}.json'.format(args.array_type, args.source_domain)

    with open(os.path.join(output_directory, out_name), 'w') as f:
        json.dump(output, f)

    print('Output files have been saved to the folder {}.'.format(output_directory))

if __name__ == "__main__":
    main()

