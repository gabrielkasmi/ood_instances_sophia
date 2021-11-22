# The effect of PV array instances and images backgrounds on OOD generalization
<b> G Kasmi, L. Dubus, P. Blanc, Y-M. Saint-Drenan </b>

<b> Presented as a poster at SophIA Summit 2021 </b> 

Material for the poster "how do solar array characteristics and images backgrounds affect OOD generalization" presented at [SophIA summit](https://univ-cotedazur.eu/events/sophia-summit).

# Table of contents

## Overview

The repository is organized in six folders :

- The folder [`dataset`](https://github.com/gabrielkasmi/ood_instances_sophia/tree/main/dataset) contains the scripts, the array masks and example images to replicate the synthetic dataset used for the experiments.
- The folder ood_scores contains the necessary material to replicate the data that is used to compute the figure 2 of the poster "F1 scores on the OOD datasets"
- The folder heatmap contains the necessary material to replicate the data used to generate the heatmap (figure 3) of the poster "OOD F1 scores for different (background,instance) combinations"
- The folder dimensionality_plots contains the material to compute plots of the dimensionality estimates using [Islam et. al. (2021)](https://arxiv.org/abs/2101.11604) dimensionality estimation technique.
- The folder [`misc`](https://github.com/gabrielkasmi/ood_instances_sophia/tree/main/misc) contains miscellaneous scripts and notebooks that have been used to compute the sanity checks and folder [`misc/data`](https://github.com/gabrielkasmi/ood_instances_sophia/tree/main/misc/data). This folder contains the files used to generate the results of the poster.
- The folder [`utils`](https://github.com/gabrielkasmi/ood_instances_sophia/tree/main/utils) contains utility functions that are needed to run the scripts.

## Set up and usage

- Please note that for the scripts to work, the paths in the files should be replaced by your own paths. 
- Due to size constraints, only samples are provided for the background images. The complete folders are avaiable on request. 
- The file `ood_instance.yml` allows you to create the environment `ood_instance` by typing `conda env create -f ood_instance.yml` from the CLI. 
- For each part of the paper, scripts have been executed to generate raw data. Based on this raw data, figures have been generated in dedicated notebooks. 


# Motivation

# Synthetic dataset 

# Results

# References 

Islam, M. A., Kowal, M., Esser, P., Jia, S., Ommer, B., Derpanis, K. G., & Bruce, N. (2021). Shape or texture: Understanding discriminative features in cnns. arXiv preprint [arXiv:2101.11604](https://arxiv.org/abs/2101.11604).

Nagarajan, V., Andreassen, A., & Neyshabur, B. (2020). Understanding the failure modes of out-of-distribution generalization. arXiv preprint [arXiv:2010.15775](https://arxiv.org/abs/2010.15775).

Wang, R., Camilo, J., Collins, L. M., Bradbury, K., & Malof, J. M. (2017, October). The poor generalization of deep convolutional networks to aerial imagery from new geographic locations: an empirical study with solar array detection. In [2017 IEEE Applied Imagery Pattern Recognition Workshop (AIPR) (pp. 1-8). IEEE](https://ieeexplore.ieee.org/document/8457965).
