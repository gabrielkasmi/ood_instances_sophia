# -*- coding: utf-8 -*-

import numpy as np 

# Companion functions

def filter_dimensions(dimensions, ndims = 3):
    """
    given a list of dimensions [array, background, residual]
    returns three arrays with nan filtered
    """
    if ndims == 3:
        # extract the values
        array_dimensions = np.array([d[0] for d in dimensions])
        background_dimensions = np.array([d[1] for d in dimensions])
        residual_dimension = np.array([d[2] for d in dimensions])


        # filter the nans
        array_dimensions = array_dimensions[np.logical_not(np.isnan(array_dimensions))]
        background_dimensions = background_dimensions[np.logical_not(np.isnan(background_dimensions))]
        residual_dimension = residual_dimension[np.logical_not(np.isnan(residual_dimension))]

        return array_dimensions, background_dimensions, residual_dimension
    elif ndims == 2:
        # extract the values
        factor_dimension = np.array([d[0] for d in dimensions])
        residual_dimension = np.array([d[1] for d in dimensions])

        # filter the nans
        factor_dimension = factor_dimension[np.logical_not(np.isnan(factor_dimension))]
        residual_dimension = residual_dimension[np.logical_not(np.isnan(residual_dimension))]

        return factor_dimension, residual_dimension
    