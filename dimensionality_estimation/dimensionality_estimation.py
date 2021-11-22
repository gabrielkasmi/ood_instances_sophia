# -*- coding: utf-8 -*-

import numpy as np
import numpy as np
from src import custom_dataloaders
import os
import json
import torch
from torch.utils.data import DataLoader
import random
import tqdm
import torch.nn as nn
from torchvision.models import Inception3, resnet50
import argparse
import yaml

random.seed(42)
torch.manual_seed(42)


"""
In this file, we estimate the dimensionality of latent factors using the method provided by
Islam et al (ICLR, 2021). 

This script takes as input arguments :


the path/to/model
the path/to/outputs
the path/to/dataset

the batch size
the array type 
the background type

the number of factors
the residual index 
device

the bootstrap size : number of repetitions to approximate a distribution

"""

# Arguments
parser = argparse.ArgumentParser(description = 'Dimensionnality estimation')

parser.add_argument('--device', default = 'cuda:0', help = "GPU device")
parser.add_argument('--batch_size', default = 512, help = "Batch size", type=int)

parser.add_argument('--array_type', default = None, help = 'Instance of arrays to consider. If None, all instances are considered.')
parser.add_argument('--background_type', default = None, help = 'Instance of background to consider. If None, all instances are considered.')

parser.add_argument('--bootstrap', default = 100, help = 'Number of Monte Carlo replications to perform.', type = int)

parser.add_argument('--output_type', default = None, help = 'Which part of the paper to compute.')
parser.add_argument('--model_type', default = None, help = 'if one wants to do the computations for a specific model.')



args = parser.parse_args()

# Load the configuration file
config = 'config.yml'

with open(config, 'rb') as f:
    configuration = yaml.load(f, Loader=yaml.FullLoader)

# Retrieve the directories and parameters
output_directory = configuration.get("output_directory", 'results') 
figs_dir = configuration.get("figs_dir") 
dataset_dir = configuration.get("dataset_dir") 
model_family = configuration.get('model', 'inception')

inception_directory = configuration.get('inception_directory')
resnet_init_directory = configuration.get('resnet_init_directory')
resnet_pretrained_directory = configuration.get('resnet_pre_trained_directory')

n_factors = configuration.get('n_factors', 3)
residual_index = configuration.get('residual_index', 2)

isSanity = configuration.get('isSanity', False)
isReality = configuration.get('isReality', False)

# Set up the directories

if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

if not os.path.isdir(figs_dir):
    os.mkdir(figs_dir)


# main functions 
# setting up the model and the dataset

def model_setup(model_family, model_directory, device):
    model = torch.load(model_directory)
    model.fc = nn.Identity()
    model.to(device)
    return model

model_directories = {
    'inception': inception_directory,
    'resnet_random' : resnet_init_directory,
    'resnet_pretrained' : resnet_pretrained_directory
}
# Set up the dataset


dataset_directories = { # if we are not evaluating on real data, set up the directories
    'inception' : os.path.join(dataset_dir, 'PITW_299/test'),
    'resnet_random' : os.path.join(dataset_dir, 'PITW_224/test'),
    'resnet_pretrained' : os.path.join(dataset_dir, 'PITW_224/test'),
    'reality' : '/data/GabrielKasmi/data/ign/dataset_d034'
}

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
    Dimensionality estimation function. Reused from Islam et al (ICLR 2021)
    Licenced under MIT Licence

    Copyright (c) 2021 Md Amirul Islam 

    Link to the repository : https://github.com/islamamirul/shape_texture_neuron 
    Link to the script in which the function is defined : https://github.com/islamamirul/shape_texture_neuron/blob/main/dim_estimation/utils.py

    Estimates the dimensionality of the factors. 

    Modification to the source code : 
    Added a check that returns a nan if the sampling is such that
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

# function that create a desired part of the input

def table_1():
    """
    computes the table 1. Table 1 gives the estimated dimensionality 
    for the factors array and background. The dimensionality is computed
    bootstrap time for each three model.

    returns a dictionnary where the key is the model and 
    the values are a list of list of estimated dimensionalities.
    """

    results = {}

    if args.model_type is None: # Check if a model is specified by the user

        models = ['inception', 'resnet_pretrained', 'resnet_random']

    else:
        models = [args.model_type]

    for model_name in models:

        results[model_name] = []

        print('Estimating with model {} ...'.format(model_name))

        # set up the model and the data

        model = torch.load(model_directories[model_name])
        model.fc = nn.Identity()
        model.to(args.device)
        print('model loaded')

        data_array, data_background = data_setup(dataset_directories[model_name], args.array_type, args.background_type)

        # data has been set up. 
        # table 1 focuses on the overall estimations
        # we compute b times the representation and the associated dimensionality

        for _ in tqdm.tqdm(range(args.bootstrap)):

            # compute the representation
            output, factor_list = compute_representations(data_array, data_background, model, args, factor_lb = 0, factor_ub = 1)

            # estimate the dimensionality 
            dims, _ = dim_est(output, factor_list, n_factors, residual_index)

            # save the results
            results[model_name].append(dims)


        print('Done with model {}.'.format(model_name))

    print('Computation completed.')

    return results


def table_2():
    """
    computes the table 2. Table 2 gives the estimated dimensionality 
    for the factor array only. The dimensionality is computed
    bootstrap times for each three model.

    returns a dictionnary where the key is the model and 
    the values are a list of list of estimated dimensionalities.
    """

    results = {}

    results = {}

    if args.model_type is None: # Check if a model is specified by the user

        models = ['inception', 'resnet_pretrained', 'resnet_random']

    else:
        models = [args.model_type]

    for model_name in models:

        results[model_name] = []

        print('Estimating with model {} ...'.format(model_name))

        # set up the model and the data

        model = torch.load(model_directories[model_name])
        model.fc = nn.Identity()
        model.to(args.device)
        print('model loaded')

        data_array, data_background = data_setup(dataset_directories[model_name], args.array_type, args.background_type)

        # data has been set up. 
        # table 1 focuses on the overall estimations
        # we compute b times the representation and the associated dimensionality

        for _ in tqdm.tqdm(range(args.bootstrap)):

            # compute the representation
            output, factor_list = compute_representations(data_array, data_background, model, args, factor_lb = 0, factor_ub = 0)

            # estimate the dimensionality 
            dims, _ = dim_est(output, factor_list, 2, 1)

            # save the results
            results[model_name].append(dims)


        print('Done with model {}.'.format(model_name))

    print('Computation completed.')

    return results


def table_3():
    """
    computes the table 3. Table 3 gives the estimated dimensionality 
    for the factor array only. The dimensionality is computed
    bootstrap times for each three model.

    returns a dictionnary where the key is the model and 
    the values are a list of list of estimated dimensionalities.
    """

    results = {}
    results = {}

    if args.model_type is None: # Check if a model is specified by the user

        models = ['inception', 'resnet_pretrained', 'resnet_random']

    else:
        models = [args.model_type]

    for model_name in models:

        results[model_name] = []

        print('Estimating with model {} ...'.format(model_name))

        # set up the model and the data

        model = torch.load(model_directories[model_name])
        model.fc = nn.Identity()
        model.to(args.device)
        print('model loaded')

        data_array, data_background = data_setup(dataset_directories[model_name], args.array_type, args.background_type)

        # data has been set up. 
        # table 1 focuses on the overall estimations
        # we compute b times the representation and the associated dimensionality

        for _ in tqdm.tqdm(range(args.bootstrap)):

            # compute the representation
            output, factor_list = compute_representations(data_background, data_array, model, args, factor_lb = 0, factor_ub = 0) # reverse the datasets

            # estimate the dimensionality 
            dims, _ = dim_est(output, factor_list, 2, 1)

            # save the results
            results[model_name].append(dims)


        print('Done with model {}.'.format(model_name))

    print('Computation completed.')

    return results




def figure_1():
    """
    computes the figure 1
    """

    # dictionnary

    results = {}

    # Get the models

    # cases to consider
    # we estimate the dimensionality for the following instances
    cases = ['LB', 'LN', 'SB', 'SN', 'large', 'small', 'black', "blue"]

    results = {}

    if args.model_type is None: # Check if a model is specified by the user

        models = ['inception', 'resnet_pretrained', 'resnet_random']

    else:
        models = [args.model_type]

    for case in cases:


        results[case]= {}

        for model_name in models:

            print('Estimating case {} with model {}...'.format(case, model_name))

            results[case][model_name] = []

            # Setting up the model 

            model = model_setup(model_name, model_directories[model_name], args.device)
            # Setting up the dataloader
            data_array, data_background = data_setup(dataset_directories[model_name], case, args.background_type)

            for _ in tqdm.tqdm(range(args.bootstrap)):

                # compute the representation
                output, factor_list = compute_representations(data_array, data_background, model, args, factor_lb = 0, factor_ub = 1)

                # estimate the dimensionality 
                dims, _ = dim_est(output, factor_list, n_factors, residual_index)

                # save the results
                results[case][model_name].append(dims)
            # Empty cache once the computations are completed.
            torch.cuda.empty_cache()


    print('Case {} completed.'.format(case))

    return results


def figure_2():
    """
    computes the figure 2
    """

    # dictionnary

    results = {}

    # Get the models

    # cases to consider
    # we estimate the dimensionality for the following instances
    cases = ['forest', 'field']

    results = {}

    if args.model_type is None: # Check if a model is specified by the user

        models = ['inception', 'resnet_pretrained', 'resnet_random']

    else:
        models = [args.model_type]

    for case in cases:


        results[case]= {}

        for model_name in models:

            print('Estimating case {} with model {}...'.format(case, model_name))

            results[case][model_name] = []

            # Setting up the model 

            model = model_setup(model_name, model_directories[model_name], args.device)
            # Setting up the dataloader
            data_array, data_background = data_setup(dataset_directories[model_name], args.array_type, case)

            for _ in tqdm.tqdm(range(args.bootstrap)):

                # compute the representation
                output, factor_list = compute_representations(data_array, data_background, model, args, factor_lb = 0, factor_ub = 1)



                # estimate the dimensionality 
                dims, _ = dim_est(output, factor_list, n_factors, residual_index)

                # save the results
                results[case][model_name].append(dims)

    print('Case {} completed.'.format(case))

    return results

def reality_check():
    """
    performs the reality check, which consists, for the three models, 
    to estimate the dimensionality in three cases : 
    - array/background
    - array only
    - background only

    returns the output dictionnary

    """


    # dictionnary

    results = {}

    # Get the models

    # cases to consider
    # we estimate the dimensionality for the following instances
    cases = ['array', 'background', 'mixed']

    cases_factor = { # dictionnary that stores the uppoer bounds and lower bounds for the factors when 
                     # computing the representations;
        'array' : (0,0),
        'background' : (0,0),
        'mixed' : (0,1)
    }

    models = ['inception'] # no evaluation with resnet because the corresponding dataset does not yet exist (11/10/21)

    for case in cases:


        results[case]= {}

        factor_lb, factor_ub = cases_factor[case]

        # Setting up the dataloader
        # corresponds to the real data
        data_array, data_background = data_setup(dataset_directories['reality'], None, None, isSanity = False, isReality = True)

        for model_name in models:

            print('Starting case {} with model {}...'.format(case, model_name))

            results[case][model_name] = []

            # Setting up the model 

            model = model_setup(model_name, model_directories[model_name], args.device)

            for _ in tqdm.tqdm(range(args.bootstrap)):

                # compute the representation
                # we swap the background and arrays in the inputs if we only compute the background
                if case == 'background':
                    output, factor_list = compute_representations(data_background, data_array, model, args, factor_lb = factor_lb, factor_ub = factor_ub)
                else:
                    output, factor_list = compute_representations(data_array, data_background, model, args, factor_lb = factor_lb, factor_ub = factor_ub)


                # estimate the dimensionality 
                if case in ['background', 'array']:
                    dims, _ = dim_est(output, factor_list, 2, 1)
                else:
                    dims, _ = dim_est(output, factor_list, n_factors, residual_index)

                # save the results
                results[case][model_name].append(dims)

    print('Case {} completed.'.format(case))

    return results

def sanity_check():
    """
    performs the sanity check, which consists, for the three models, 
    to estimate the dimensionality for two meaningless factors.

    returns the output dictionnary

    """
    # dictionnary

    results = {}

    # Get the models

    models = ['inception', 'resnet_pretrained', 'resnet_random']

    for model_name in models:

        print('Starting sanity check with model {}...'.format(model_name))

        results[model_name] = []

        # Setting up the model 

        model = model_setup(model_name, model_directories[model_name], args.device)
        data_array, data_background = data_setup(dataset_directories[model_name], None, None, isSanity = True)


        for _ in tqdm.tqdm(range(args.bootstrap)):

            # compute the representation
            output, factor_list = compute_representations(data_array, data_background, model, args, factor_lb = 0, factor_ub = 1)

            # estimate the dimensionality 
            dims, _ = dim_est(output, factor_list, n_factors, residual_index)

            # save the results
            results[model_name].append(dims)

    return results


if __name__ == "__main__":

    # compute the desired output

    if args.output_type is None:
        print('Error, no output inputed.')
        raise ValueError

    if args.output_type == 'table_1':
        output = table_1()

    if args.output_type == 'table_2':
        output = table_2()

    if args.output_type == 'table_3':
        output = table_3()

    if args.output_type == 'tables': 
        print('Computing table 1...')
        output_1 = table_1()
        torch.cuda.empty_cache()
        print('Computing table 2...')
        output_2 = table_2()
        torch.cuda.empty_cache()
        print('Computing table 3...')
        output_3 = table_3()
        
        print('Computations complete.')


    if args.output_type == 'figures': # Preferably with batch_size = 256
        print('Computing figure 1...')
        output_1 = figure_1()
        torch.cuda.empty_cache()
        print('Computing figure 2...')
        output_2 = figure_2()
        torch.cuda.empty_cache()
        print('Computations complete.')


    if args.output_type == 'figure_1': # Preferably with batch_size = 256
        output = figure_1()

    if args.output_type == 'figure_2':
        output = figure_2()

    if args.output_type == 'reality':
        output = reality_check()

    if args.output_type == 'sanity':
        output = sanity_check()


    if args.output_type == 'checks':  # Preferably with batch_size = 256 
        print('Computing sanity check...')
        output_1 = sanity_check()
        torch.cuda.empty_cache()
        print('Computing reality check...')
        output_2 = reality_check()  # Preferably with batch_size = 
        torch.cuda.empty_cache()
        print('Computations complete.')



    # Export the results

    if args.output_type in ['table_1', 'table_2', 'table_3', 'figure_1', 'figure_2', 'sanity', 'reality']: # case of a single item

        if args.model_type is None: # Check whether we additionnally store a table for a specific model
            out_name = args.output_type + '.json'
        else:
            out_name = args.output_type + '_' + args.model_type + ".json"

        with open(os.path.join(output_directory, out_name), 'w') as f:
            json.dump(output, f)

    elif args.output_type == 'tables':
        # save in three dedicated json files

        out_name_1, out_name_2, out_name_3 = "table_1.json", "table_2.json", "table_3.json"

        outputs = {
            out_name_1 : output_1,
            out_name_2 : output_2,
            out_name_3 : output_3
        }

        for output in outputs.keys():
            with open(os.path.join(output_directory, output), 'w') as f:
                json.dump(outputs[output], f)

    elif args.output_type == 'figures': #same, but for the figures
        
        out_name_1, out_name_2,  = "figure_1.json", "figure_2.json"

        outputs = {
            out_name_1 : output_1,
            out_name_2 : output_2,
        }

        for output in outputs.keys():
            with open(os.path.join(output_directory, output), 'w') as f:
                json.dump(outputs[output], f)
        
    elif args.output_type == 'checks': # same, but for the sanity and reality checks

        out_name_1, out_name_2,  = "sanity_check.json", "reality_check.json"

        outputs = {
            out_name_1 : output_1,
            out_name_2 : output_2,
        }

        for output in outputs.keys():
            with open(os.path.join(output_directory, output), 'w') as f:
                json.dump(outputs[output], f)
