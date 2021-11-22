# -*- coding: utf-8 -*-

# Libraries
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random

seed = 42

random.seed(seed)

# Contains custom dataset classes designed to estimate
# the mutual information between factors

def read_specified_label_file(label_file, label):
    """
    Reads a label file and returns the rows there the label
    is specified. E.g. returns all positive labels if 1 is inputed
    and returns 0 otherwise.
    """     
    dataframe = pd.read_csv(label_file, header = None, names = ["img", "label"])

    return dataframe[dataframe['label'] == label]


class MutualArrayInformation(Dataset):
    """
    Picks an array of a given type and returns a pair of the same type, but with a different background
    type.
    """
    def __init__(self, img_dir, array_type = None):
        """
        Args:
            img_dir (string): directory with all the images.
            type : the type of array relative to which the MI is computed.
                - possible array types : ['LB', 'SB', 'LB', 'SB', 'black', 'blue', 'large', 'small'] 
                - possible background types : ['forest', 'fields']
        """


        # Correspondance between the inputed type and the corresponding label files.
        # One will pick an image from one file and its pair in the other.
        labels_correspondance = {
            'blue' : ['blue_field_labels.csv', 'blue_forest_labels.csv'],
            'black' : ['black_field_labels.csv', 'black_forest_labels.csv'],

            'large' : ['forest_large_labels.csv', 'field_large_labels.csv'],
            'small' : ['forest_small_labels.csv', 'field_small_labels.csv'],

            'SN' : ['SN_fields_labels.csv', 'SN_forest_labels.csv'],
            'SB' : ['SB_fields_labels.csv', 'SB_forest_labels.csv'],

            'LN' : ['LN_fields_labels.csv', 'LN_forest_labels.csv'],
            'LB' : ['LB_fields_labels.csv', 'LB_forest_labels.csv'],

            'array' : ['fields_labels.csv', 'forest_labels.csv']
        }

        # Get the label files of the inputed type

        array_type = array_type if array_type is not None else 'array' # if no type is imputed, will consider arrays as a whole

        labels_img_1, labels_img_2 = labels_correspondance[array_type]

        labels_1 = os.path.join(img_dir, labels_img_1)
        labels_2 = os.path.join(img_dir, labels_img_2)

        # Get the files
        self.img_labels_1 = read_specified_label_file(labels_1, 1) 
        self.img_labels_2 = read_specified_label_file(labels_2, 1).sample(frac = 1, random_state = seed).reset_index(drop = True)

        self.img_dir = img_dir


    def __len__(self):
        return len(self.img_labels_1) 

    def _normalize(self, image, mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]):
        transform = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        return transform(image)

    def __getitem__(self, idx):
        """
        For a randomly picked array image, returns an array of the same type but 
        with a different background
        """

        img_1_path = os.path.join(self.img_dir, self.img_labels_1.iloc[idx, 0])
        img_2_path = os.path.join(self.img_dir, self.img_labels_2.iloc[idx, 0])

        attribution = random.sample([img_1_path, img_2_path], 2) # randomize which background is picked up first.

        image_1 = transforms.ToTensor()(Image.open(attribution[0]).convert("RGB"))

        image_2 = transforms.ToTensor()(Image.open(attribution[1]).convert("RGB"))

        return self._normalize(image_1), self._normalize(image_2)

class MutualBackgroundInformation(Dataset):
    """
    Picks an array of a given type and returns a pair of the same type, but with a different background
    type.
    """
    def __init__(self, img_dir, background_type = None):
        """
        Args:
            img_dir (string): directory with all the images.
            type : the type of array relative to which the MI is computed.
                - possible array types : ['LB', 'SB', 'LB', 'SB', 'black', 'blue', 'large', 'small'] 
                - possible background types : ['forest', 'fields']
        """


        # Correspondance between the inputed type and the corresponding label files.
        # One will pick an image from one file and its pair in the other.
        labels_correspondance = {
            'field' : ['SN_fields_labels.csv', 'LN_fields_labels.csv', 'SB_fields_labels.csv', 'LB_fields_labels.csv'],
            'forest' : ['SN_forest_labels.csv', 'LN_forest_labels.csv', 'SB_forest_labels.csv', 'LB_forest_labels.csv'],

            'background' : ['black_large_labels.csv', 'black_small_labels.csv', 'blue_large_labels.csv', 'blue_small_labels.csv']
        }

        background_type = background_type if background_type is not None else 'background' # Inputs a generic type if no type is specified.

        self.reference_labels = labels_correspondance[background_type]
        self.img_dir = img_dir

    def __len__(self):
        # All frames have the same lenght, so we can open anyone of them

        test_frame = self.reference_labels[0]
        
        return len(pd.read_csv(os.path.join(self.img_dir, test_frame), header = None))

    def _normalize(self, image, mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]):
        transform = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        return transform(image)

    def __getitem__(self, idx):
        """
        We pick an image in the reference file. 
        For a randomly picked array image, returns an array of the same type but 
        with a different background
        """

        # Randomly define a reference label

        # Pick two labels from the reference labels.
        # One will be the reference, the other the variant
        labels = random.sample(self.reference_labels, 2) 

        reference_label = pd.read_csv(os.path.join(self.img_dir, labels[0]), header = None) # Open the dataframe
        variant_label = pd.read_csv(os.path.join(self.img_dir, labels[1]), header = None).sample(frac = 1, random_state = seed).reset_index(drop = True)


        reference_image_path = os.path.join(self.img_dir, reference_label.iloc[idx, 0]) # Pick an image
        variant_image_path = os.path.join(self.img_dir, variant_label.iloc[idx, 0]) # Pick an image

        image_1 = transforms.ToTensor()(Image.open(reference_image_path).convert("RGB"))
        image_2 = transforms.ToTensor()(Image.open(variant_image_path).convert("RGB"))
    
        return self._normalize(image_1), self._normalize(image_2)


class DummyInformation(Dataset):
    """
    Sanity check. 

    Computes the mutual information between two random from the dataset.

    """
    def __init__(self, img_dir):
        """
        Args:
            img_dir (string): directory with all the images.
            type : the type of array relative to which the MI is computed.
                - possible array types : ['LB', 'SB', 'LB', 'SB', 'black', 'blue', 'large', 'small'] 
                - possible background types : ['forest', 'fields']
        """


        # Correspondance between the inputed type and the corresponding label files.
        # One will pick an image from one file and its pair in the other.
        labels_correspondance = {
            'blue' : ['blue_field_labels.csv', 'blue_forest_labels.csv'],
            'black' : ['black_field_labels.csv', 'black_forest_labels.csv'],

            'large' : ['forest_large_labels.csv', 'field_large_labels.csv'],
            'small' : ['forest_small_labels.csv', 'field_small_labels.csv'],

            'SN' : ['SN_fields_labels.csv', 'SN_forest_labels.csv'],
            'SB' : ['SB_fields_labels.csv', 'SB_forest_labels.csv'],

            'LN' : ['LN_fields_labels.csv', 'LN_forest_labels.csv'],
            'LB' : ['LB_fields_labels.csv', 'LB_forest_labels.csv'],

            'array' : ['labels.csv', 'labels.csv']
        }

        # Get the label files of the inputed type

        array_type = 'array' # if no type is imputed, will consider arrays as a whole

        labels_img_1, labels_img_2 = labels_correspondance[array_type]

        labels_1 = os.path.join(img_dir, labels_img_1)
        labels_2 = os.path.join(img_dir, labels_img_2)

        # Get the files
        # Image 1 an 2 are randomly picked.
        self.img_labels_1 = pd.read_csv(labels_1, header = None).sample(frac = 1, random_state = seed)#read_specified_label_file(labels_1, 1) 
        self.img_labels_2 = pd.read_csv(labels_2, header = None).sample(frac = 1, random_state = seed) #read_specified_label_file(labels_2, 1).sample(frac = 1, random_state = seed).reset_index(drop = True)

        self.img_dir = img_dir


    def __len__(self):
        return len(self.img_labels_1) 

    def _normalize(self, image, mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]):
        transform = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        return transform(image)

    def __getitem__(self, idx):
        """
        For a randomly picked array image, returns an array of the same type but 
        with a different background
        """

        img_1_path = os.path.join(self.img_dir, self.img_labels_1.iloc[idx, 0])
        img_2_path = os.path.join(self.img_dir, self.img_labels_2.iloc[idx, 0])

        attribution = random.sample([img_1_path, img_2_path], 2) # randomize which background is picked up first.

        image_1 = transforms.ToTensor()(Image.open(attribution[0]).convert("RGB"))

        image_2 = transforms.ToTensor()(Image.open(attribution[1]).convert("RGB"))

        return self._normalize(image_1), self._normalize(image_2)



class MutualRealInformation(Dataset):
    """
    Picks two images of arrays
    """
    def __init__(self, img_dir, factor):
        """
        Args:
            img_dir (string): directory with all the images.
            factor : 0 or 1, the type of factor to be inputed.
        """


        # Correspondance between the inputed type and the corresponding label files.
        # One will pick an image from one file and its pair in the other.

        self.factor = factor

        # Get the label files of the inputed type

        labels_1 = os.path.join(img_dir, 'labels.csv')
        labels_2 = os.path.join(img_dir, 'labels.csv')

        # Get the files

        # For the array : picks two images of arrays
        # For the background, picks to images of backgrounds 
        self.img_labels_1 = read_specified_label_file(labels_1, self.factor) 
        self.img_labels_2 = read_specified_label_file(labels_2, self.factor).sample(frac = 1, random_state = seed).reset_index(drop = True)

        self.img_dir = img_dir


    def __len__(self):
        return len(self.img_labels_1) 

    def _normalize(self, image, mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]):
        transform = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        return transform(image)

    def __getitem__(self, idx):
        """
        For a randomly picked array image, returns an array of the same type but 
        with a different background
        """

        img_1_path = os.path.join(self.img_dir, self.img_labels_1.iloc[idx, 0])
        img_2_path = os.path.join(self.img_dir, self.img_labels_2.iloc[idx, 0])

        attribution = random.sample([img_1_path, img_2_path], 2) # randomize which background is picked up first.

        image_1 = transforms.ToTensor()(Image.open(attribution[0]).convert("RGB"))

        image_2 = transforms.ToTensor()(Image.open(attribution[1]).convert("RGB"))

        return self._normalize(image_1), self._normalize(image_2)


"""
companion functions
"""


def data_setup(dataset_dir, array_type, background_type, isSanity = False, isReality = False):
    """
    sets up the data loaders
    """

    # Sanity check
    if isSanity:
        array = custom_dataloaders.DummyInformation(dataset_dir)
        background = custom_dataloaders.DummyInformation(dataset_dir)

    # Reality check1
    elif isReality:
        array = custom_dataloaders.MutualRealInformation(dataset_dir, factor = 1)
        background = custom_dataloaders.MutualRealInformation(dataset_dir, factor = 0)

    else:        
        array = custom_dataloaders.MutualArrayInformation(dataset_dir, array_type = array_type)
        background = custom_dataloaders.MutualBackgroundInformation(dataset_dir, background_type = background_type)

    data_array = DataLoader(array, batch_size = args.batch_size, shuffle = True)
    data_background = DataLoader(background, batch_size = args.batch_size, shuffle = True)

    return data_array, data_background

def compute_representations(data_array, data_background, model, args ,factor_lb = 0, factor_ub = 1):
    """
    Given a model, a number of factors and two data loaders, 
    computes the representations and store them in a dictionnary

    factor_ub, factor_lb : the upper bound and lower bound to sample the factors from

    returns this dictionnary and the factor list
    """
    factor_list = []
    outputs = {'examples1': [],
               'examples2': []}

    model = model.eval()


    for item_array, item_background in zip(data_array, data_background):
        
        # Pick which factor is to be estimated
        factor = random.randint(factor_lb, factor_ub)
        
        pairs = {0 : item_array, 1 : item_background}
        
        # Unwrap the pairs and pass them trough the model
        imagesA, imagesB = pairs[factor]
        
        with torch.no_grad():

            imagesA = imagesA.to(args.device)
            imagesB = imagesB.to(args.device)
        
            output1 = model(imagesA)
            output2 = model(imagesB)

        # add factor and output to list / array for processing dimensions later on
        factor_list.append(factor)
        outputs['examples1'].append(output1.detach().cpu().numpy())
        outputs['examples2'].append(output2.detach().cpu().numpy())       


    return outputs, factor_list

def dim_est(output_dict, factor_list, n_factors, residual_index):
    """
    Taken from from Islam et al (ICLR 2021)
    Estimates the dimensionality of the factors. 

    simply added a check that returns a nan if the sampling is such that
    one of the factors is empty.
    """
    # grab flattened factors, examples
    # factors = data_out.labels["factor"]
    # za = data_out.labels["example1"].squeeze()
    # zb = data_out.labels["example2"].squeeze()

    # factors = np.random.choice(2, 21845) # shape=21845
    # za = np.random.rand(21845, 2048)
    # zb = np.random.rand(21845, 2048)

    za = np.concatenate(output_dict['examples1'])
    zb = np.concatenate(output_dict['examples2'])
    factors = np.array(factor_list)


    za_by_factor = dict()
    zb_by_factor = dict()
    mean_by_factor = dict()
    score_by_factor = dict()
    individual_scores = dict()

    zall = np.concatenate([za,zb], 0)
    mean = np.mean(zall, 0, keepdims=True)

    # za_means = np.mean(za,axis=1)
    # zb_means = np.mean(zb,axis=1)
    # za_vars = np.mean((za - za_means[:, None]) * (za - za_means[:, None]), 1)
    # zb_vars = np.mean((za - zb_means[:, None]) * (za - zb_means[:, None]), 1)

    var = np.sum(np.mean((zall-mean)*(zall-mean), 0))
    for f in range(n_factors):
        if f != residual_index:
            indices = np.where(factors==f)[0]

            ### sanity check
            # it is possible that during the computations, a factor is not assigned any value. 
            # in this case we directly return a nan in order to go to the next iteration.

            if len(indices) == 0:
                print("Factor {} has not been assigned any representations. Ending the computations for the current iteration.".format(f))

                return [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]

            za_by_factor[f] = za[indices]
            zb_by_factor[f] = zb[indices]
            mean_by_factor[f] = 0.5*(np.mean(za_by_factor[f], 0, keepdims=True)+np.mean(zb_by_factor[f], 0, keepdims=True))
            # score_by_factor[f] = np.sum(np.mean(np.abs((za_by_factor[f] - mean_by_factor[f]) * (zb_by_factor[f] - mean_by_factor[f])), 0))
            # score_by_factor[f] = score_by_factor[f] / var
            # OG
            score_by_factor[f] = np.sum(np.mean((za_by_factor[f]-mean_by_factor[f])*(zb_by_factor[f]-mean_by_factor[f]), 0))
            score_by_factor[f] = score_by_factor[f]/var
            idv = np.mean((za_by_factor[f]-mean_by_factor[f])*(zb_by_factor[f]-mean_by_factor[f]), 0)/var
            individual_scores[f] = idv
        #   new method
        #     score_by_factor[f] = np.abs(np.mean((za_by_factor[f] - mean_by_factor[f]) * (zb_by_factor[f] - mean_by_factor[f]), 0))
        #     score_by_factor[f] = np.sum(score_by_factor[f])
        #     score_by_factor[f] = score_by_factor[f] / var

            # new with threshhold
            # sigmoid
            # score_by_factor[f] = sigmoid(score_by_factor[f])
            # score_by_factor[f] = np.abs(np.mean((za_by_factor[f] - mean_by_factor[f]) * (zb_by_factor[f] - mean_by_factor[f]), 0))
            # score_by_factor[f] = score_by_factor[f] / var
            # score_by_factor[f] = np.where(score_by_factor[f] > 0.5, 1.0, 0.0 )
            # score_by_factor[f] = np.sum(score_by_factor[f])
        else:
            # individual_scores[f] = np.ones(za_by_factor[0].shape[0])
            score_by_factor[f] = 1.0


    scores = np.array([score_by_factor[f] for f in range(n_factors)])
    
    # SOFTMAX
    m = np.max(scores)
    e = np.exp(scores-m)
    softmaxed = e / np.sum(e)
    dim = za.shape[1]
    try : 
        dims = [int(s*dim) for s in softmaxed]
     
        dims[-1] = dim - sum(dims[:-1])
        dims_percent = dims.copy()
        for i in range(len(dims)):
            dims_percent[i] = round(100*(dims[i] / sum(dims)),1)
    except :
        dims = [np.nan, np.nan, np.nan]
        dims_percent = [np.nan, np.nan, np.nan]
    return dims, dims_percent

