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

The synthetic dataset consists in two domains. A source domain, comprised of 80,000 samples with 4 array types and 2 background types and a target domain comprised of 4 array types (different from those of the source domain) and one background type. Each domain is splitted into training, validation and testing datasets. We also include to intermediate domain testing datasets, one containing source backgrounds and target arrays, and the other source arrays and the target background. 

These images come from IGN aerial images, that are accessible [here]() and provided under open license.

Samples from the background are depicted below : 

<p align="center">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/dataset/backgrounds/fields/34-2018-0680-6245-LA93-0M20-E080_(10913.5%2C%2024667.5).png" width="200">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/dataset/backgrounds/forest/34-2018-0700-6250-LA93-0M20-E080_(8222.5%2C%204933.5).png" width="200">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/dataset/backgrounds/urban/34-2018-0690-6250-LA93-0M20-E080_(12109.5, 22275.5).png" width="200">
</p>

And samples from the in domain (leftmost and center left images) and out domain arrays (center right and rightmost images). 

<p align="center">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/dataset/arrays/in_domain_arrays/LB_mask.png" width="200">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/dataset/arrays/in_domain_arrays/SN_mask.png" width="200">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/dataset/arrays/ood_arrays/ood_1_mask.png" width="200">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/dataset/arrays/ood_arrays/ood_2_mask.png" width="200">
</p>

For each sample, the creation procedure is as follows : 
- Apply random rotation, displacements and symmetries to the array and random rotation and symmetries to the background
- With probability 1/2, apply the array on the image to generate a positively labelled image, otherwise leave the background as is to generate a negative sample. 

The dataset is balanced, for one positive sample, one negative sample is generated. Each subgroup of data is also evenly represented. More precisely, the source domain includes 8 (array, instances) pairs and for each pair, we generate the same number of samples (positive and negatives). Moreover, we generate label files for each of the subgroups, as well as for group of subgroups in order to be able to train the model on subsamples of the training dataset only. 

The target domain includes 4 (array, instances) pairs and for each pair, we generate the same number of samples. Intermediate domains are also balanced in terms of positive samples and type of arrays and backgrounds.


# Results

## OOD performance

We first decompose the out-of-domain error into two components:
- The error due to the fact that the model has to identify solar array instances that it has never seen before,
- The error due to the fact that the model has to identify solar arrays over unseen backgrounds.

To isolate these effects, we evaluate OOD performance in three settings:
- 1. We consider new solar array instances but in domain backgrounds (lefmost boxplot)
- 2. In-domain solar array instances but new backgroun (boxplot in the middle of the figure below)
- 3. Out-of-domain array instances and backgrounds (rightmost boxtplot)

We can see that the change in backgrounds drives the OOD error. Put otherwise, according to our experiment, if a model fails to generalize well out-of-domain, it is mostly due to the fact that out-of-domain samples depict unseen backgrounds. A possible explanation for this phenomenon is that when facing small objects such as solar arrays, detection models will heavily rely on background features to make their prediction. However, as recalled by [Gulrajani and Lopez-Paz (2020)](https://arxiv.org/abs/2007.01434) or [Nagarajan et. al. (2020)](https://arxiv.org/abs/2010.15775), features extracted from the background of the image are <i> spurious </i> features in the sense that they are likely to change when one is shifted from one domain to the other. 

In order to further inspect the impact of the background and the type of solar array instance on OOD performance, we perform a second experiment where the target dataset remains fixed and the composition of the training dataset changes. Our hope is to show some backgrounds and some solar array instances can allow for a better OOD generalization than others.

<p align="center">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/figs/display/F1_score_in_domain_to_ood.png" width="250">
</p>

## The impact of the source domain on OOD performance

We now consider the reverse phenomenon and set a fixed OOD dataset with unseen array instances and an unseen background. The composition of the training dataset on the other hand varies : it contains more or less large or blue arrays (y-axis) and more or less images drawn over the fields background (x-axis). Each cell outputs the average F1 score of the model on the OOD dataset, given a fixed share of (background, array instance) in the training dataset. 

We can see that the composition of the training dataset has an important impact on performance. Moreover, the final performance is more affected by the background than by the solar array instance. This conforts the idea according to which some background characteristics prevent the model from learning too many spurious features during training. In our case, a plausible explanation is that arrays are more contrasted on fields backgrounds than on forest backgrounds and therefore making the distinction between the background (which is irrelevant) and the foreground (i.e. the solar array) more explicit.

<p align="center">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/figs/display/heatmap_large_fields_scores.png" width="250">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/figs/display/heatmap_blue_fields_scores.png" width="250">
</p>


## Dimensionality estimates as an explanation for OOD performance

Finally, we want to see whether it is possible to quantify the semantic concepts (namely solar array and background) that are learned during training in the latent representation of the model and use it as a predictor for OOD performance. The idea would be that the larger the dimensionality in the latent representation, the more detailed the representation of the semantic concept. Then, for the backgrounds, the smaller the dimensionality the better the OOD performance and for arrays, the larger the dimensionality the better the OOD generalization. 
### Methodology

We apply the methodology proposed by [Islam et al (2021)](https://arxiv.org/abs/2101.11604) to estimate the dimension of the semantic factors "solar array" and "backgrounds" in the representation computed by the model. The starting point is the method proposed by [Esser et. al. (2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Esser_A_Disentangling_Invertible_Interpretation_Network_for_Explaining_Latent_Representations_CVPR_2020_paper.html) for explaining latent representation. More details on the methodology can be found in the working paper [`ood_generalization_wp.pdf`](https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/ood_generalization_wp.pdf).

### Results

As it can be seen from the figure below, our results are inconclusive. All instances of solar arrays have the same estimated dimensionality (around 400) and all types of backgrounds also have the same dimensionality estimation (around 400 as well). As such, based on these results, it is not possible to say that there is a correlation between the dimensionality of the instance and how it is suited for OOD generalisation.

<p align="center">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/figs/display/dimensionality_estimates.png" width="250">
</p>

## Sanity checks 

In addition to the results rapported above, we conduct several sanity checks in order to see whether the dimensionality estimation measures are sensical. We first see how the dimensionality estiamtion varies when one factor is omitted. On the upper figure below, we estimate the dimensionality of the arrays only, then of the background only and finaly of both factors. We can see that the orders of magnitude remain the same, no matter whether the dimensionality of the two factors are estimated or only one. 

Besides, we also apply the methodology on real data and see that the estimated dimensionalities are of the same magnitude than in the experimental setting (leftmost). Both sanity checks have been done on three models, the Inception v3 model from [Rausch et. al (2020)](https://arxiv.org/abs/2012.03690) and a ResNet50, one with pretraining on ImageNet and the other with random initialization. All models are fined tuned on our synthetic dataset before the dimensionality estimation is carried out. 

These sanity checks highlight the fact that the dimensionality estimate is indeed well correlated with the mutual information between the two images of interest and that these estimates are not model dependent. Additional sanity checks are reported in the appendix of the working paper [`ood_generalization_wp.pdf`](https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/ood_generalization_wp.pdf).

<p align="center">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/figs/display/reality_check_plot.png" width="500">
<img src="https://github.com/gabrielkasmi/ood_instances_sophia/blob/main/figs/display/sanity_check_plot.png" width="500">
</p>

# Summary and future work

This experiment shows that for small object detection on overhead imagery, OOD performance is mostly affected by the background characteristics. A possible explanation is that some backgrounds allows for a better disentanglement between  <i> predictive </i> (i.e. correlated with the semantic label one wants to predict) and <i> spurious </i> (i.e. correlated with the training dataset) features. 

Future work should therefore forcus on consolidating this claim in a more principled framework. To this end, it is necessary to take into account additional factors that can vary from one dataset to another such as the image characteristics (ground sampling distance, brightness, projection of the ground on the image). It is also necessary to show that on "good" backgrounds, the model does indeed extract <i> predictive </i> features.

# References 

Islam, M. A., Kowal, M., Esser, P., Jia, S., Ommer, B., Derpanis, K. G., & Bruce, N. (2021). Shape or texture: Understanding discriminative features in cnns. arXiv preprint [arXiv:2101.11604](https://arxiv.org/abs/2101.11604).

Nagarajan, V., Andreassen, A., & Neyshabur, B. (2020). Understanding the failure modes of out-of-distribution generalization. arXiv preprint [arXiv:2010.15775](https://arxiv.org/abs/2010.15775).

Wang, R., Camilo, J., Collins, L. M., Bradbury, K., & Malof, J. M. (2017, October). The poor generalization of deep convolutional networks to aerial imagery from new geographic locations: an empirical study with solar array detection. In [2017 IEEE Applied Imagery Pattern Recognition Workshop (AIPR) (pp. 1-8). IEEE](https://ieeexplore.ieee.org/document/8457965).

Cooper, A., Boix, X., Harari, D., Madan, S., Pfister, H., Sasaki, T., & Sinha, P. (2021). To Which Out-Of-Distribution Object Orientations Are DNNs Capable of Generalizing?. arXiv preprint [arXiv:2109.13445](https://arxiv.org/abs/2109.13445). 

Esser, P., Rombach, R., & Ommer, B. (2020). A disentangling invertible interpretation network for explaining latent representations. In [Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9223-9232)](https://openaccess.thecvf.com/content_CVPR_2020/html/Esser_A_Disentangling_Invertible_Interpretation_Network_for_Explaining_Latent_Representations_CVPR_2020_paper.html).

Gulrajani, I., & Lopez-Paz, D. (2020). In search of lost domain generalization. arXiv preprint [arXiv:2007.01434](https://arxiv.org/abs/2007.01434).

Rausch, B., Mayer, K., Arlt, M. L., Gust, G., Staudt, P., Weinhardt, C., ... & Rajagopal, R. (2020). An Enriched Automated PV Registry: Combining Image Recognition and 3D Building Data. arXiv preprint [arXiv:2012.03690](https://arxiv.org/abs/2012.03690).
