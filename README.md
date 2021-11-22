# The effect of PV array instances and images backgrounds on OOD generalization
<b> G Kasmi, L. Dubus, P. Blanc, Y-M. Saint-Drenan </b>

<b> Presented as a poster at SophIA Summit 2021 </b> 

Material for the poster "The effect of PV array instances and images backgrounds on OOD generalization" presented at [SophIA summit](https://univ-cotedazur.eu/events/sophia-summit) (17-19 november 2021)

# Table of contents

We provide an overview and a notice regarding the usage of the repository. A brief description of the work and the resutls is provided after this description.

## Overview

The repository is organized in seven folders :

- The folder [`dataset`](https://github.com/gabrielkasmi/ood_instances_sophia/tree/main/dataset) contains the scripts, the array masks and example images to replicate the synthetic dataset used for the experiments.
- The folder [`ood_performance`](https://github.com/gabrielkasmi/ood_instances_sophia/tree/main/ood_performance) contains the necessary material to replicate the data that is used to compute the figure 2 of the poster "F1 scores on the OOD datasets"
- The folder [`heatmap`](https://github.com/gabrielkasmi/ood_instances_sophia/tree/main/heatmap) contains the necessary material to replicate the data used to generate the heatmap (figure 3) of the poster "OOD F1 scores for different (background,instance) combinations"
- The folder [`dimensionality_estimation`](https://github.com/gabrielkasmi/ood_instances_sophia/tree/main/dimensionality_estimation) contains the material to compute plots of the dimensionality estimates using [Islam et. al. (2021)](https://arxiv.org/abs/2101.11604) dimensionality estimation technique.
- The folder [`misc`](https://github.com/gabrielkasmi/ood_instances_sophia/tree/main/misc) contains additional material and the folder [`misc/data`](https://github.com/gabrielkasmi/ood_instances_sophia/tree/main/misc/data). This folder contains the files used to generate the results of the poster.
- The folder [`utils`](https://github.com/gabrielkasmi/ood_instances_sophia/tree/main/utils) contains utility functions that are needed to run the scripts.
- The folder [`figs`](https://github.com/gabrielkasmi/ood_instances_sophia/tree/main/figs) contains the output figures.

## Set up and usage

- Please note that for the scripts to work, the paths in source scripts and the `config.yml` configuration file should be replaced by your own paths. Moreover, it may be necessary to create output directories.
- Due to size constraints, only samples are provided for the background images. The complete folders are avaiable on request. 
- The file `ood_instance.yml` allows you to create the environment `ood_instance` by typing `conda env create -f ood_instance.yml` from the CLI. 
- For each part of the paper, scripts have been executed to generate raw data. Based on this raw data, figures have been generated in dedicated notebooks. 

## Models 

For the main experiments (OOD performance, heatmap and dimensionality estimation), the model used is a ResNet 50 which can be directly downloaded from PyTorch. Model weights are available on request.

# Motivation and objectives

Deep learning based models for remote sensing of solar arrays often experience an impredictible performance drop when deployed to a new location ([Wang et. al. (2017)](https://ieeexplore.ieee.org/document/8457965)). This problem is caused by the fact that machine learning methods struggle to generalize out-of-domain (OOD). In this poster, we design an experimental setup that aims at disentangling the impact of the background and the solar array type on OOD performance. 

The setup consists in a synthetic dataset that mixes different types of solar arrays (which we call "instances") and different types of background. We leverage this synthetic dataset to study OOD generalization in two directions : 
- In the first case, we consider a fixed source dataset and see whether a model fails to generalize to new instances or new backgrounds
- In the second case, we consider a fixed target dataset and see whether OOD generalization can be affected by the composition of the source dataset

In order to provide insights on the uneven ability to generalize, we leverage [Islam et al (2021)](https://arxiv.org/abs/2101.11604) dimensionality estimation technique to see whether depending on the instance and background, the number of dimensions in the latent that encode the semantic factors "solar array" and "background" varies.

The questions we wish to address are the following : 

- <b> Is the failure to generalize predominantly due to unseen arrays or unseen backgrounds ? </b>
- <b> Has the type of background or array instance an influence on OOD performance ? </b>
- <b> Can we quantify which types of backgrounds or solar arrays are better for generalization ? </b>

# Synthetic dataset 



# Results

## OOD performance

<p align="center">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/figs/display/F1_score_in_domain_to_ood.png" width="250">
</p>

## The impact of the source domain on OOD performance

## Dimensionality estimates as an explanation for OOD performance

### Methodology

We apply the methodology proposed by [Islam et al (2021)](https://arxiv.org/abs/2101.11604) to estimate the dimension of the semantic factors "solar array" and "backgrounds" in the representation computed by the model. The starting point is the method proposed by [Esser et. al. (2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Esser_A_Disentangling_Invertible_Interpretation_Network_for_Explaining_Latent_Representations_CVPR_2020_paper.html) for explaining latent representation. More details on the methodology can be found in the working papier `ood_generalization_wp.pdf` available in this repository.

### Results

## Sanity checks 

In addition to the results rapported above, we conduct several sanity checks in order to see whether the dimensionality estimation measures are sensical. 

# References 

Islam, M. A., Kowal, M., Esser, P., Jia, S., Ommer, B., Derpanis, K. G., & Bruce, N. (2021). Shape or texture: Understanding discriminative features in cnns. arXiv preprint [arXiv:2101.11604](https://arxiv.org/abs/2101.11604).

Nagarajan, V., Andreassen, A., & Neyshabur, B. (2020). Understanding the failure modes of out-of-distribution generalization. arXiv preprint [arXiv:2010.15775](https://arxiv.org/abs/2010.15775).

Wang, R., Camilo, J., Collins, L. M., Bradbury, K., & Malof, J. M. (2017, October). The poor generalization of deep convolutional networks to aerial imagery from new geographic locations: an empirical study with solar array detection. In [2017 IEEE Applied Imagery Pattern Recognition Workshop (AIPR) (pp. 1-8). IEEE](https://ieeexplore.ieee.org/document/8457965).

Cooper, A., Boix, X., Harari, D., Madan, S., Pfister, H., Sasaki, T., & Sinha, P. (2021). To Which Out-Of-Distribution Object Orientations Are DNNs Capable of Generalizing?. arXiv preprint [arXiv:2109.13445](https://arxiv.org/abs/2109.13445). 

Esser, P., Rombach, R., & Ommer, B. (2020). A disentangling invertible interpretation network for explaining latent representations. In [Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9223-9232)](https://openaccess.thecvf.com/content_CVPR_2020/html/Esser_A_Disentangling_Invertible_Interpretation_Network_for_Explaining_Latent_Representations_CVPR_2020_paper.html).

