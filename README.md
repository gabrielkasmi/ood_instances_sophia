# The effect of PV array instances and images backgrounds on OOD generalization
## G Kasmi, L. Dubus, P. Blanc, Y-M. Saint-Drenan
### Presented as a poster at SophIA Summit 2021

Material for the poster "how do solar array characteristics and images backgrounds affect OOD generalization" presented at [SophIA summit](https://univ-cotedazur.eu/events/sophia-summit).

# Table of contents

## Overview

The repository is organized in six folders :

- The folder [`dataset`](https://github.com/gabrielkasmi/ood_instances_sophia/tree/main/dataset) contains the scripts, the array masks and example images to replicate the synthetic dataset used for the experiments.
- The folder ood_scores contains the necessary material to replicate the data that is used to compute the figure 2 of the poster "F1 scores on the OOD datasets"
- The folder heatmap contains the necessary material to replicate the data used to generate the heatmap (figure 3) of the poster "OOD F1 scores for different (background,instance) combinations"
- The folder dimensionality_plots contains the material to compute plots of the dimensionality estimates using [Islam et. al. (2021)](https://arxiv.org/abs/2101.11604) dimensionality estimation technique.
- The folder misc contains miscellaneous scripts and notebooks that have been used to compute the sanity checks and folder `misc/data`. This folder contains the files used to generate the results of the poster.
- The folder `utils` contains utility functions that are needed to run the scripts.

## Set up and usage

- Please note that for the scripts to work, the paths in the files should be replaced by your own paths. 
- Due to size constraints, only samples are provided for the background images. The complete folders are avaiable on request. 
- The file `ood_instance.yml` allows you to create the environment `ood_instance` by typing `conda` from the CLI. 
- For each part of the paper, scripts have been executed to generate raw data. Based on this raw data, figures have been generated in dedicated notebooks. 


# Motivation

# Synthetic dataset 

# Results

# References 

Islam, M. A., Kowal, M., Esser, P., Jia, S., Ommer, B., Derpanis, K. G., & Bruce, N. (2021). Shape or texture: Understanding discriminative features in cnns. arXiv preprint [arXiv:2101.11604](https://arxiv.org/abs/2101.11604).
