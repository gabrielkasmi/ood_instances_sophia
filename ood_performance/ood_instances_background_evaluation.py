# -*- coding: utf-8 -*-

"""
Trains a model on the source dataset
Then evaluates it on three OOD datasets:
- OOD instances
- OOD backgrounds
- OOD instances and background

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
parser.add_argument('--max_count', default = None, help = 'Maximal number of samples to sample from.', type = int)
parser.add_argument('--source_domain', default = None, help = "The source domain. The target domain will be deduced.")

parser.add_argument('--bootstrap', default = 50, help = "Number of bootstrap iterations to estimate the dimensionality", type = int)
parser.add_argument('--n_iter', default = 10, help = "Number of train/dimensionality estimation/test iterations", type = int)

args = parser.parse_args()


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

# Data

# Load the data

# Load the correct image size
if model_family == 'resnet':
    dataset_dir = os.path.join(data_dir, 'PITW_224')
elif model_family == 'inception':
    dataset_dir = os.path.join(data_dir, 'PITW_299')

training_path = os.path.join(dataset_dir, 'train')
annotations_train = os.path.join(training_path, "labels.csv")

validation_path = os.path.join(dataset_dir, 'validation')
annotations_validation = os.path.join(validation_path, 'labels.csv')

test_path = os.path.join(dataset_dir, 'test')
annotations_test = os.path.join(test_path, 'labels.csv')

ood_background = os.path.join(dataset_dir, "test_ood_background")
annotations_ood_background = os.path.join(ood_background, 'labels.csv')

ood_instances = os.path.join(dataset_dir, "test_ood_instances")
annotations_ood_instances = os.path.join(ood_instances, 'labels.csv')

# ood dataset : custom path 
ood_dir = 'PITW_urban'
ood = os.path.join(ood_dir, "test")
annotations_ood = os.path.join(ood, 'labels.csv')

training_data = dataset.BDPVClassificationDataset(annotations_train, training_path, transform = transforms, max_count = args.max_count)
validation_data = dataset.BDPVClassificationDataset(annotations_validation, validation_path, transform = transforms, max_count = args.max_count)
test_data = dataset.BDPVClassificationDataset(annotations_test, test_path, transform = transforms, max_count = args.max_count)

ood_background_data = dataset.BDPVClassificationDataset(annotations_ood_background, ood_background, transform = transforms, max_count = args.max_count)
ood_instances_data = dataset.BDPVClassificationDataset(annotations_ood_instances, ood_instances, transform = transforms, max_count = args.max_count)

ood_data = dataset.BDPVClassificationDataset(annotations_ood, ood, transform = transforms, max_count = args.max_count)

# Initialize the data

train = DataLoader(training_data, batch_size = args.batch_size, shuffle = True)
val = DataLoader(validation_data, batch_size = args.batch_size, shuffle = True)  
test = DataLoader(test_data, batch_size = args.batch_size, shuffle = True)  

background_ood = DataLoader(ood_background_data, batch_size = args.batch_size, shuffle = True)
instances_ood = DataLoader(ood_instances_data, batch_size = args.batch_size, shuffle = True)
ood_dataset = DataLoader(ood_data, batch_size = args.batch_size, shuffle = True)



dataloader = {"train": train, 'val' : val, 'test' : test} 
ood_dataloader = {'ood_background' : background_ood,
                  'ood_instances' : instances_ood,
                  'ood' : ood_dataset}

def train(dataloader):
    """
    trains the model
    """

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

def ood_evaluation(best_model, dataloader):
    """
    evaluates the model on the target domain. Takes the best model as input.

    """
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

    ood_outputs = {}

    for case in dataloader.keys():
        print('Now evaluating the model for the case {}...'.format(case))

        probabilities = []
        all_labels = []

        with torch.no_grad():


            for data in tqdm(dataloader[case]):
                # i +=1
                images, labels, _ = data

                # move the images to the device
                images = images.to(args.device)

                #labels = labels.detach().cpu().numpy()

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

        # save the results

        ood_outputs[case] = {}

        ood_outputs[case]['precision'] = list(precision)
        ood_outputs[case]['recall'] = list(recall)
        ood_outputs[case]['f1'] = list(f1)
        ood_outputs[case]['best_threshold'] = best_threshold
    
    return ood_outputs


def main():

    output = {}

    for i in range(args.n_iter):

        print('Starting iteration {}...'.format(i))

        output[str(i)] = {}

        # train the model
        print('Training the model')

        train_outputs, best_model = train(dataloader)

        output[str(i)]['in_domain'] = train_outputs

        print('Model has been trained. Now evaluating the model on the ood dataset')

        # evaluate performance on the target domain
        ood_outputs = ood_evaluation(best_model, ood_dataloader)

        output[str(i)]['out_domain'] = ood_outputs

        print("OOD performance evaluated.")

        # save the outputs in a dedicated json file

    out_name = 'ood_instance_and_background_evaluation.json'.format(args.array_type, args.source_domain)

    with open(os.path.join(output_directory, out_name), 'w') as f:
        json.dump(output, f)

    print('Output files have been saved to the folder {}.'.format(output_directory))

if __name__ == "__main__":
    main()

