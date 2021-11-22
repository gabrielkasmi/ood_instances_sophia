# -*- coding: utf-8 -*-

# Libraries
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from sklearn.utils import shuffle


# Class for the BDPV classification dataset.
# Classification because the labels are binary
# Indicating whether or not the image contains an array.
# The images contained in the dataset are RBG images.

class BDPVClassificationDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None, max_count = None):
        """
        Args:
            annotations_file (string): name of the csv file with labels
            img_dir (string): directory with all the images.
            transform (callable, optional): optional transform to be applied on a sample.
        """
        self.img_labels = shuffle(pd.read_csv(annotations_file, header = None), random_state = 42)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.max_count = max_count

    def __len__(self):
        if self.max_count is None:
            return len(self.img_labels)
        else: 
            return min(self.max_count, len(self.img_labels))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        label = self.img_labels.iloc[idx, 1]
        name = self.img_labels.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        return image, label, name

    # From here : custom attributes

    def image_name(self, idx):
        return self.img_labels.iloc[idx, 0]
    
    def labels(self):
        self.img_labels.columns = ["img_name", "label"]
        return self.img_labels


# Function that performs a train/test split and 
# returns a dictionnary of dataloader instances.

def train_test_split(dataset_dir, transforms, batch_size, validation_split = 0.2, shuffle = True, seed = 42):
    """Performs a train/test split"""

    # Get the annotation file in the folder
    annotations_file = os.path.join(dataset_dir , 'labels.csv')

    # Load the dataset
    data = BDPVClassificationDataset(annotations_file, dataset_dir, transform = transforms)

    # Indices for training and validation
    dataset_size = data.__len__()
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle :
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    # Split the indices
    train_indices, val_indices = indices[split:], indices[:split]

    # create data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, sampler = train_sampler)
    validation_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, sampler = valid_sampler)

    # Return the dictionnary
    return {'train' : train_loader, 'val' : validation_loader}

# Function that performs a train/val/test split and 
# returns a dictionnary of dataloader instances.

def train_val_test_split(dataset_dir, transforms, batch_size, validation_split = 0.15, test_split = 0.15, shuffle = True, seed = 42):
    """Performs a train/val/test split"""

    # Get the annotation file in the folder
    annotations_file = os.path.join(dataset_dir, 'labels.csv')

    # Load the dataset
    data = BDPVClassificationDataset(annotations_file, dataset_dir, transform = transforms)

    # Indices for training and validation
    dataset_size = data.__len__()
    indices = list(range(dataset_size))
    split_val = int(np.floor(validation_split * dataset_size))
    split_test = int(np.floor(test_split * dataset_size))

    if shuffle : # shuffle indices
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    # Split the indices
    train_indices, val_indices, test_indices = indices[split_val + split_test:], indices[:split_val], indices[split_val:split_val + split_test]

    # create data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, sampler = train_sampler)
    validation_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, sampler = valid_sampler)
    test_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, sampler = test_sampler)

    # Return the dictionnary
    return {'train' : train_loader, 'val' : validation_loader, 'test' : test_loader}

# Function that creates a custom label files with desired proportions of different instances

def set_up_label_file(dataset_dir, background_prop, ref_array_prop, array_type, type, seed = 42):
    """
    creates a label file

    returns the path to the corresponding label file in the corresponding folder
    the label file has the corresponding characteristics in terms of number of instances from
    each category

    args : 
    dataset_dir : path/to/dataset
    background_prop : the share of images coming from the field background
    ref_array_prop : the share of arrays from the reference instance
    array_type : the reference array type
    type : either train, val or test.

    """

    # target directory : corresponds to dataset/dir/test, dataset/dir/train or dataset/dir/val
    target_directory = os.path.join(dataset_dir, type)
    
    # Compute the number of instances required per category
    # according to the following table 
    # and given the proportions passed as input
    # the main difficulty is that the number of instances is bounded 
    # (e.g. max 4000 field/forest in the validation dataset)
    # so we need to find TT such that xx <= x_ub and YY, ZZ 
    # match the desired proportions.
    # 
    # 
    #       | field | forest | total
    # ------------------------------
    # good  |   xx  |   xx   |  YY
    # ------------------------------
    # bad   |   xx  |   xx   |  YY
    # ------------------------------
    # total |   ZZ  |   ZZ   |  TT, tt = 1   
    #

    # load the dataframes

    fields_labels = pd.read_csv(os.path.join(target_directory, 'fields_labels.csv'), header = None)
    forest_labels = pd.read_csv(os.path.join(target_directory, 'forest_labels.csv'), header = None)

    # compute the instance_size, i.e. the number of elements per instance. By definition, 
    # it is equal to the total length of the label dataframe / 4 (instances here are groups 
    # of two primary instances)

    if type == "test":
        instance_size = 4000
    elif type == 'train':
        instance_size = 20000
    elif type == "validation":
        instance_size = 2000


    # set up the frequencies as marginals

    domain_marginals = np.array([background_prop, 1 - background_prop])
    instance_marginals = np.array([ref_array_prop, 1 - ref_array_prop])

    # matrix of the frequencies
    instance_frequencies = np.outer(instance_marginals, domain_marginals)

    # all instances are bounded by the same value, the total size of the 
    # dataset is determined by the highest frequency

    n_items = int(instance_size / (np.max(instance_frequencies)))

    # we get the number of samples per instances such that the frequencies
    # are satisfied and the upper bound on the number of instances is also
    # satisfied
    instance_counts = np.around((instance_frequencies * n_items)).astype(int)

    # the total sample size will vary between the number of samples in one
    # instances (if alpha = beta = 0 or 1) and the number of samples if 
    # alpha = beta = 1/2. 

    # now sample for each instance the number of needed

    if array_type == 'large':
        
        # extract each cell of the frequency matrix
        
        count_field_large = instance_counts[0,0]
        count_field_small = instance_counts[1,0]
        
        count_forest_large = instance_counts[0,1]
        count_forest_small = instance_counts[1,1]

        # extract the instances from the dataframes
        large_fields = fields_labels.iloc[[i[0] == 'L' for i in fields_labels[0]]].sample(count_field_large, random_state = seed)
        small_fields = fields_labels.iloc[[i[0] == 'S' for i in fields_labels[0]]].sample(count_field_small, random_state = seed)
        
        large_forest = forest_labels.iloc[[i[0] == 'L' for i in forest_labels[0]]].sample(count_forest_large, random_state = seed)
        small_forest = forest_labels.iloc[[i[0] == 'S' for i in forest_labels[0]]].sample(count_forest_small, random_state = seed)
        
        concatenated_data_frame = pd.concat([large_fields, small_fields, large_forest, small_forest])
        
    elif array_type == 'blue':
        
        # extract each cell of the frequency matrix
        
        count_field_blue = instance_counts[0,0]
        count_field_black = instance_counts[1,0]
        
        count_forest_blue = instance_counts[0,1]
        count_forest_black = instance_counts[1,1]

        # extract the instances from the dataframes
        blue_fields = fields_labels.iloc[[i[1] == 'B' for i in fields_labels[0]]].sample(count_field_blue, random_state = seed)
        black_fields = fields_labels.iloc[[i[1] == 'N' for i in fields_labels[0]]].sample(count_field_black, random_state = seed)
        
        blue_forest = forest_labels.iloc[[i[1] == 'B' for i in forest_labels[0]]].sample(count_forest_blue, random_state = seed)
        black_forest = forest_labels.iloc[[i[1] == 'N' for i in forest_labels[0]]].sample(count_forest_black, random_state = seed)
        
        concatenated_data_frame = pd.concat([blue_fields, black_fields, blue_forest, black_forest])#, ignore_index = True)


        
    else:
        print('Reference array type is not "blue" or "large". Please pass a correct value')
        raise ValueError
   

    label_name = 'labels_ref_array_{}_prop_{}_field_prop_{}.csv'.format(array_type, str(ref_array_prop), str(background_prop))
    
    # export the dataframe with the desired name to the target location (in train/val/test)
    concatenated_data_frame.to_csv(os.path.join(target_directory,label_name), header = None, index = False)

    return os.path.join(target_directory, label_name)