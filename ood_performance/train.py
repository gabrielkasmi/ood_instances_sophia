# -*- coding: utf-8 -*-

"""
Trains a model.

TODO. Support for TensorBoard

Args : 
- device : the device on which the model should be trained
- n_epochs : the number of epochs
- print_every ; after how many steps the model should be evaluated on the validation dataset
- name : the name of the model (for saving the figures and the model)
- batch_size : the batch size

Rests on the config.yml file for the source and target directories 
The dataset directory should have the following structure :

path/to/dataset/
        train/
        validation/
        test/

"""

import torch
import numpy as np
import PIL
import random
import os
from src import dataset, helpers
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
dataset_dir = configuration.get("dataset_dir") 

# Retrieve the options for the model
pretrained : configuration.get('pretrained', True)
model_family = configuration.get('model', 'inception')
model_weight = configuration.get('model_weight')
image_size = configuration.get('image_size')
label_name = configuration.get('label_name', 'labels.csv')

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
parser.add_argument('--print_every', default = 50, help = "Evaluate the model every print_every iterations.", type=int)
parser.add_argument('--name', default = 'model', help = "Name of the model.")

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
        torchvision.transforms.CenterCrop(224),
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

def main():

    # Data

    training_path = os.path.join(dataset_dir, 'train')
    annotations_train = os.path.join(training_path, label_name)
    
    validation_path = os.path.join(dataset_dir, 'validation')
    annotations_validation = os.path.join(validation_path, label_name)

    test_path = os.path.join(dataset_dir, 'test')
    annotations_test = os.path.join(test_path, label_name)

    training_data = dataset.BDPVClassificationDataset(annotations_train, training_path, transform = transforms)
    validation_data = dataset.BDPVClassificationDataset(annotations_validation, validation_path, transform = transforms)
    test_data = dataset.BDPVClassificationDataset(annotations_test, test_path, transform = transforms)

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
        model = torchvision.models.resnet50(pretrained = False)

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
                
                if waiting == 5:
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

    # Plot and export

    def ticks(step, i):
        return step + i * step

    def x_axis(step, values):
        return [ticks(step, i) for i in range(len(values))]


    plt.plot(x_axis(args.print_every, train_losses), train_losses, label = "Training Loss")
    plt.plot(x_axis(args.print_every, test_losses), test_losses, label = 'Validation Loss')

    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.legend()
    training_plot = best_model_name[:-4] + '_train_val_losses.png'
    plt.savefig(os.path.join(figs_dir, training_plot))

    # Fine tune the classification threshold
    print('Model trained. Now fine tuning the classification threshold.')


    # Thresholds to be considered
    results_models = {}
    thresholds = np.linspace(0.01,.99, 99)

    model.eval()

    # Forward pass on the validation dataset and accumulate the probabilities 
    # in a single vector.

    probabilities = []
    all_labels = []

    with torch.no_grad():

        for data in tqdm(dataloader["val"]):
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

    # Save the dictionnary of results
    results_dictionnary = best_model_name[:-4] + "_threshold_results.json"

    # Copy the results as string for the json export
    results = {}

    for threshold in thresholds:
        results[str(threshold)] = {}
        results[str(threshold)]['true_positives'] = str(results_models[threshold]['true_positives'])
        results[str(threshold)]['false_positives'] = str(results_models[threshold]['false_positives'])
        results[str(threshold)]['true_negatives'] = str(results_models[threshold]['true_negatives'])
        results[str(threshold)]['false_negatives'] = str(results_models[threshold]['false_negatives'])
        results[str(threshold)]['accuracy'] = str(results_models[threshold]['accuracy'])


    with open(os.path.join(files_dir, results_dictionnary), "w") as f: # save as a json file
        json.dump(results, f)

    
    # Compute the precision and recall
    precision, recall = helpers.compute_pr_curve(results_models)

    # Compute the F1 score
    f1 = return_f1(precision, recall)

    # Determine the best treshold (that maximizes F1) and the corresponding precision and recall

    best_threshold = thresholds[np.argmax(f1)]
    best_precision, best_recall = precision[np.argmax(f1)], recall[np.argmax(f1)]


    # Plot and save
    plot_precision, plot_recall = plot_curves(precision, recall)

    plt.clf() # Clear the figure 


    plt.plot(plot_precision, plot_recall)
    plt.scatter(best_precision, best_recall, c = 'red')

    pr_fig = best_model_name[:-4] + '_precision_recall_plot.png'

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.savefig(os.path.join(figs_dir,pr_fig))

    print('Threshold fined tuned. Now evaluating the model on the test set.')

    # Finally, evaluate the optimized model on the test set.
    model.eval()

    with torch.no_grad():

        true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0
        total = 0

        for inputs, labels, _ in tqdm(dataloader["test"]):

            labels = labels.to(torch.float32)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model.forward(inputs)
            outputs = F.softmax(outputs, dim=1) # the model returns the unnormalized probs. Softmax it to get probs
            outputs = outputs[:,1]
            batch_loss = criterion(outputs, labels)
            test_loss += batch_loss.item()
            predicted = (outputs >= best_threshold).long()  # return 0 or 1 if an array has been detected

            # compute the accuracy of the classification

            tp, fp, tn, fn = helpers.confusion(predicted, labels)
            true_positives += tp
            false_positives += fp
            true_negatives += tn
            false_negatives += fn
    
    # Compute the F1 score
    precision = np.divide(true_positives, (true_positives + false_positives))
    recall = np.divide(true_positives, (true_positives + false_negatives))
    
    f1 = 2 * np.divide((precision * recall), (precision + recall))


    print("The best model {} has been saved in the directory '{}'. \n The best threshold is {:.2f} and corresponding metrics on the test set are : \n F1 : {:.3f} \n Precision {:.3f} \n Recall {:.3f}".format(best_model_name, models_dir, best_threshold, f1, precision, recall))
    print('Training complete.')


if __name__ == "__main__":
    main()