# -*- coding: utf-8 -*-

"""
SYNTHETIC EXPERIMENT 
We train a model on a specific composition of the PITW dataset.

Composition include : 
- a proportion between 0% and 100% [step 25%] of field vs forest
- a proportion between 0% and 100% of 'good' vs 'bad' array [step 25%], 
  either blue or large for the "good" instance

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

# set up a custom subfolder to save the results
results_directory = os.path.join(output_directory, 'synthetic_experiment')

# create a directory if it does not already exist.
if not os.path.isdir(results_directory):
    os.mkdir(results_directory)

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
parser.add_argument('--array_type', default = None, help = "Reference class of array.") 
# parser.add_argument('--max_count', default = 20000, help = 'Maximal number of samples to sample from.', type = int)
parser.add_argument('--domain', default = 'fields', help = "The domain to consider. OOD instances will be drawn from the same domain.")
parser.add_argument('--n_iter', default = 5, help = "Number of train/dimensionality estimation/test iterations", type = int)

parser.add_argument("--ref_array_prop", default = None, help = "Proportion of the reference class of array. ", type = float)

args = parser.parse_args()

if args.array_type is None:
    print('Input a value for the reference array type')
    raise ValueError

if args.ref_array_prop is None:
    print('Input a value for the reference array type.')
    raise ValueError

def set_up_training_data(background_prop, ref_array_prop, array_type, seed = seed):

    """
    sets up the dataloader for the training of the model. 

    the field prop corresponds to the share of images with background field in the 
    training data
    the ref_array_prop corresponds to the share of the reference array in the 
    training data
    the array_type corresponds to the instance of the reference array (i.e. large or blue)

    Example : if background_prop = 0.2, ref_array_prop = 0. and array_type = large then
    the data is formatted as follows : 

    20% of the images come from the field backgrond, 80% from forest
    100% of positive images depict 

    """
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


    # In the dataset_dir, set up the correct annotation files

    training_path = os.path.join(dataset_dir, 'train')
    validation_path = os.path.join(dataset_dir, 'validation')
    test_path = os.path.join(dataset_dir, 'test')

    # set up the paths to the annotations files.
    # the functions set_up_label file create the label file with the desired properties in the 
    # dataset folder and 
    annotations_train = dataset.set_up_label_file(dataset_dir, background_prop, ref_array_prop, array_type, 'train', seed = seed)
    annotations_validation = dataset.set_up_label_file(dataset_dir, background_prop, ref_array_prop, array_type, 'validation', seed = seed)
    annotations_test = dataset.set_up_label_file(dataset_dir, background_prop, ref_array_prop, array_type, 'test', seed = seed)
    

    training_data = dataset.BDPVClassificationDataset(annotations_train, training_path, transform = transforms, max_count = 20000)
    validation_data = dataset.BDPVClassificationDataset(annotations_validation, validation_path, transform = transforms, max_count = 2000)
    test_data = dataset.BDPVClassificationDataset(annotations_test, test_path, transform = transforms, max_count = 4000)

    # Initialize the data

    train = DataLoader(training_data, batch_size = args.batch_size, shuffle = True)
    val = DataLoader(validation_data, batch_size = args.batch_size, shuffle = True)  
    test = DataLoader(test_data, batch_size = args.batch_size, shuffle = True)  

    # return the dataloader

    return {"train": train, 'val' : val, 'test' : test}

def train(dataloader):

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

def ood_evaluation(best_model):
    """
    evaluates the model on the target domain. Takes the best model as input.

    takes one label at a time. 

    """

    # Load the correct image size
    if model_family == 'resnet':
        dataset_dir = 'PITW_urban'
    # is not defined for inception
    #elif model_family == 'inception':
    #    dataset_dir = os.path.join(data_dir, 'PITW_299')

    
    # set up the domains 
    

    test_path = os.path.join(dataset_dir, 'test')
    annotations_test = os.path.join(test_path, 'labels.csv')

    test_data = dataset.BDPVClassificationDataset(annotations_test, test_path, transform = None)

    # Initialize the data

    test = DataLoader(test_data, batch_size = args.batch_size, shuffle = True)  

    dataloader = {'test' : test}

    # will loop over the dataloaders

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

    background_shares = [0.,0.25,0.50,0.75,1.]

    for background_share in background_shares :

        output = {}

        # set of the data

        dataloader = set_up_training_data(background_share, args.ref_array_prop, args.array_type, seed = seed)

        print("""
        Training data set up with the following characteristics : \n
        Reference array instance : {}\n
        Share of the reference array instance : {}\n
        Share of the field background : {}
        """.format(args.array_type, args.ref_array_prop, background_share))

        for i in range(args.n_iter):

            output[str(i)] = {}

            # train the model
            train_outputs, best_model = train(dataloader)

            output[str(i)]['in_domain'] = train_outputs

            print('Model has been trained. Now evaluating the model on the ood instances')

            # evaluate performance on the OOD domain

            ood_outputs = ood_evaluation(best_model)

            print('OOD evaluation complete.')
            
            output[str(i)]['ood'] = ood_outputs

            print("OOD performance evaluated.")

        # once the reps are completed,  
        # save the outputs in a dedicated json file

        out_name = 'ood_results_reference_array_{}_share_{}_field_share_{}.json'.format(args.array_type, str(args.ref_array_prop), background_share)

        with open(os.path.join(results_directory, out_name), 'w') as f:
            json.dump(output, f)

        print('Output files have been saved to the folder {}.'.format(output_directory))

    print('Done')

if __name__ == "__main__":
    main()

