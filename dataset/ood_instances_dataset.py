# -*- coding: utf-8 -*-

from PIL import Image, ImageOps, ImageChops
import numpy as np
import pandas as pd
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
import itertools
import random
import sys
import argparse


"""
Generates a test set with 16k samples with the ood instances and the in domain backgrounds (forest, field)
"""


# Arguments
parser = argparse.ArgumentParser(description = 'Dataset creation')

# parser.add_argument('--train_count', default = 40000, help = "Number of (positive) samples in the training dataset", type = int)
# parser.add_argument('--val_count', default = 4000, help = "Number of (positive) samples in the validation dataset", type = int)
parser.add_argument('--counts', default = 8000, help = "Number of (positive) samples in the testing dataset", type = int)

parser.add_argument('--crop_background', default = True, help = "Indicate whether the background images should be cropped to fit ResNet requirements.", type = bool)

parser.add_argument('--background_directory', default = 'background_images', help = "Location of the background images")
parser.add_argument('--target_directory', default = 'PITW/PITW_224', help = "Location of the target images")
parser.add_argument('--arrays_directory', default = 'arrays_new', help = "Location of the arrays ")

args = parser.parse_args()


random.seed(42)

def transform_array(im, crop = False):
    """
    applies a series of transforms to an image
    
    returns the transformed image.
    """

    if crop:
        area = (30,30,254,254)
        im = im.crop(area)
    
    
    # dimension of the input image
    width, height = im.size
    
    # random values for the rotation and offset
    rotation = np.random.randint(360)
    x_offset = np.random.randint(width / 4)
    y_offset = np.random.randint(height / 4)
        
    # compute the offset and crop the image to avoid repeated 
    # arrays

    im = ImageChops.offset(im, x_offset, y_offset)
    im.paste((0, 0, 0, 0), (0, 0, x_offset, height))
    im.paste((0, 0, 0, 0), (0, 0, width, y_offset))
    im.convert("RGBA")
    
    # rotate the image
    im = im.rotate(rotation)    

    # Mirror and flip : 
    if np.random.uniform() < .5: 
        im = ImageOps.mirror(im)
   
    if np.random.uniform() < .5:
        im = ImageOps.flip(im)
       
    return im

"""
Transforms a background image
Applies random flips and mirrors to the image.
"""
def transform_background(img, crop = False):
    """
    randomly transforms the background
    
    returns the transformed image
    """

    if crop:
        area = (30,30,254,254)
        img = img.crop(area)
    
    rotation = np.random.randint(4)
        
    img.rotate(rotation * 45)
    
    
    # Mirror and flip : 
    if np.random.uniform() < .5: 
        img = ImageOps.mirror(img)
   
    if np.random.uniform() < .5:
        img = ImageOps.flip(img)
    
    return img

"""
Combines a random transformation of the array image and
a random transformation of the background image.
"""
def create_positive_image(background, array, crop = False):
    """
    combines an array and a background
    returns the transformmed image
    """ 
    return Image.alpha_composite(transform_background(background, crop = crop), transform_array(array, crop = crop))


"""
Generates samples for a given (background, array) pair. It also generates the associated
.csv label file. 
Overall, 2 * count samples are generated : half are positive images, the other half
negative images. By convention, positive images are numbered from 1 to count and
negative images from count + 1 to 2 * count.
"""


def generate_samples(count, target_directory, background_directory, array_directory, background_type, array_type, crop = False):
    """
    Generates a given number of samples and stores them in the target directory.
    
    args
    
    count: the number of samples to generate
    target directory : the target directory (str)
    background directory : the source directory of the background images (str)
    
    background, str, 'forest' or 'fields' correspond to the background of choice
    img_type : str, corresponds to 'LB', 'LN', 'SB', 'SN'.
    
    returns None
    
    """
    
    # dictionnary that will contain the labels
    labels = {}
    
    # list of background images
    directory = os.path.join(background_directory, background_type)
    backgrounds = os.listdir(directory)

    print("Generating images for the case {}-{}".format(array_type, background_type))

    for i in tqdm(range(count)):

        # get two background images
        # on one we will display an array, on the other we won't
        array_background, no_array_background = random.sample(backgrounds, 2)
        
        # open the images
        background = Image.open(os.path.join(directory, array_background)).convert('RGBA')
        empty = Image.open(os.path.join(directory, no_array_background)).convert('RGBA')
        
        # open the array image
        image_name = "{}_mask.png".format(array_type)        
        array = Image.open(os.path.join(array_directory, image_name))
        
        # apply transforms 
        positive_image = create_positive_image(background, array, crop = crop)
        negative_image = transform_background(empty, crop = crop)
        
        # give names to the images
        # positive images are labelled from 1 to count, negative from count + 1 to 2 * count
        name_positive = "{}_{}_{}".format(array_type, background_type, i + 1)
        name_negative = "{}_{}_{}".format(array_type, background_type, count + i + 1)
        
        # save the labels 
        labels[name_positive + '.png'] = 1
        labels[name_negative + '.png'] = 0
        
        # save the images in the target directory
        positive_image.save(os.path.join(target_directory, name_positive + '.png'))
        negative_image.save(os.path.join(target_directory, name_negative + '.png'))
    
    
    # Create the dataframe
    labels_df = pd.DataFrame.from_dict(labels, orient = 'index').reset_index()
    labels_name = "{}_{}_labels.csv".format(array_type, background_type)
    labels_df.to_csv(os.path.join(target_directory, labels_name), header = None, index = False)
    
    return None

"""
Generates a full dataset (encompassing all sub catgories files).
"""
def generate_dataset(overall_count, target_directory, background_directory, array_directory, crop = False):
    """
    Generates the full dataset given a total count and input and output directories
    """
    
    # split the overall count in 8, corresponding to the 8 subcategories
    # ood_1 to _4 - fields 
    # ood_1 to _4 - forest 

    
    count = int(overall_count / 8)
    
    # cases
    array_types = ['ood_1', 'ood_2', 'ood_3', 'ood_4']
    background_types = ['fields', 'forest']
    
    
    # compute each sub category
    for array_type, background_type in itertools.product(array_types, background_types):
                
        generate_samples(count, target_directory, background_directory, array_directory, background_type, array_type, crop = crop)
            
    # generate the label files for the upper levels of clusterization.
    # the higher level is overall (no distinction)
    # the intermediate level contains one key (location, color, size)
    # the lower level contains two keys ((location, color), (location, size), (color, size))
    # the lowest level contains the three keys (location, color, size)
    
    # get the name of the .Csv files
    label_files = []
    for file in os.listdir(os.path.join(os.getcwd(),target_directory)):
        if file.endswith(".csv"):
            label_files.append(file)
    
    # higher level : merge all .csv together
    
    frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in label_files]
    overall = pd.concat(frames).reset_index(drop = True)    
    # save the dataframe
    overall.to_csv(os.path.join(target_directory, 'labels.csv'), header = None, index = False)
    
    # intermediate level
    # we have two dataframes to generate : fields/forest
        
    # black_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in label_files if label[1] == 'N']
    # blue_frames  = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in label_files if label[1] == 'B']
    # small_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in label_files if label[0] == 'S']
    # large_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in label_files if label[0] == 'L']
    field_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in label_files if label[3:-11] == 'fields']
    forest_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in label_files if label[3:-11] == 'forest']
    
    #print('forest_frames', forest_frames)
    # same trick but with the names
    
    # black_frames_names = [label for label in label_files if label[1] == 'N']
    # blue_frames_names  = [label for label in label_files if label[1] == 'B']
    # small_frames_names = [label for label in label_files if label[0] == 'S']
    # large_frames_names = [label for label in label_files if label[0] == 'L']
    field_frames_names = [label for label in label_files if label[3:-11] == 'fields'] # pas le même nom, pas la même longueur.
    forest_frames_names = [label for label in label_files if label[3:-11] == 'forest']    
    
    dataframes = [field_frames, forest_frames]
    cases = ['fields', 'forest']
    
    # in all cases, save the cooresponding dataframe.
    # for dataframe, case in zip(dataframes, cases):
        
    #     aggregated_frame = pd.concat(dataframe).reset_index(drop = True)
        
    #    filename = '{}_labels.csv'.format(case)
        
    #    aggregated_frame.to_csv(os.path.join(target_directory, filename), header = None, index = False)
    
    # lower level
    # same-is as for the intermediate level, except that we have more combinations
    # filter with respect to the previous filters
    # black_large_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in black_frames_names if label[0] == 'L']
    # black_small_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in black_frames_names if label[0] == 'S']
    
    # blue_large_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in blue_frames_names if label[0] == 'L']
    # blue_small_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in blue_frames_names if label[0] == 'S']
    
    # black_field_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in black_frames_names if label[3:-11] == 'fields']
    # black_forest_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in black_frames_names if label[3:-11] == 'forest']
    
    # blue_field_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in blue_frames_names if label[3:-11] == 'fields']
    # blue_forest_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in blue_frames_names if label[3:-11] == 'forest']
    
    # field_large_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in field_frames_names if label[0] == 'L']
    # field_small_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in field_frames_names if label[0] == 'S']
    
    # forest_large_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in forest_frames_names if label[0] == 'L']
    # forest_small_frames = [pd.read_csv(os.path.join(target_directory, label), header = None) for label in forest_frames_names if label[0] == 'S']
    

    
    # cases = ["black_large", "black_small", "blue_large", "blue_small"]
    #        "black_field", "blue_field", "black_forest", "blue_forest",
    #         "field_large", "field_small", "forest_large", "forest_small"]
    
    # dataframes = [black_large_frames, black_small_frames, blue_large_frames, blue_small_frames]
    #               black_field_frames, blue_field_frames, black_forest_frames, blue_forest_frames,
    #               field_large_frames, field_small_frames, forest_large_frames, forest_small_frames] 
          
        
    # for dataframe, case in zip(dataframes, cases):        
        
    #     aggregated_frame = pd.concat(dataframe).reset_index(drop = True)
        
    #     filename = '{}_labels.csv'.format(case)        
        
    #     aggregated_frame.to_csv(os.path.join(target_directory, filename), header = None, index = False)
    
    return None


"""
Generates the complete dataset
"""

def generate_complete_dataset(counts, target_directory, background_directory, arrays_directory, crop = False):
    """
    Wrapper that computes the whole dataset
    """
    
    # get the total number of samples to generate
    
    for count, dataset in zip(counts, ["test_ood_instances"]) : 
        
        destination = os.path.join(target_directory, dataset)

        # create the folder if necessary
        if not os.path.isdir(destination):
            os.mkdir(destination)

        # generate the samples
        print("Generating {} samples for the {} dataset...".format(2 * count, dataset))
        generate_dataset(count, destination, background_directory, arrays_directory, crop = crop)
        print("Generation of the {} set completed.".format(dataset))
        
    print("Generation completed. Images and label files are in the folder {}".format(target_directory))
    
    return None


if __name__ == "__main__":

    # Unwrap the arguements
    counts = [args.counts] # args.train_count, args.val_count, args.test_count    
    target_directory = args.target_directory
    background_directory = args.background_directory
    arrays_directory = args.arrays_directory    

    # Check that the directories exist

    if not os.path.isdir(target_directory):
        print('Target directory {} does not exist.'.format(target_directory))
        raise ValueError

    if not os.path.isdir(background_directory):
        print('Background directory {} does not exist.'.format(background_directory))
        raise ValueError

    if not os.path.isdir(arrays_directory):
        print('Arrays directory {} does not exist.'.format(arrays_directory))
        raise ValueError

    # If all checks are passed, run the function.
    generate_complete_dataset(counts, target_directory, background_directory, arrays_directory, crop = args.crop_background)


