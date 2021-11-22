# Dataset

## Overview 

This folder contains the material for the creation of the synthetic datasets used in the poster. 

- The folder `arrays` contains the `.png` masks of the arrays that have been used to generate the in domain and out of domain datasets. 
- The folder `backgrounds` contains example images for the three backgrounds types used in the experiment : `forest`, `fields` and `urban`

The four `.py` scripts should be launched from the command line in the environment `ood_instance` and allow to construct the following datasets : 

- The script `generate_dataset_.py` constructs a dataset with training, testing and validation folders with arrays of the `in_domain_arrays` folder and backgrounds from the folders `fields` and ``forest`.
- The

## Important notice
- Due to size constraints, only one sample image per background is provided. The original dataset use 1,000 images of the instance `forest` and `field` and 353 of the instance `urban`. Full folders are available on request.
- In the four scripts, you may have to change the input and output directories in the arguments of the script for the function to work on your machine.
