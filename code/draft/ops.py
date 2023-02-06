# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:25:44 2021

@author: keira
"""

import numpy as np
import keras.backend as K
import tensorflow as tf


def cumulative(x):
    '''

    Parameters
    ----------
    x : variable values(s) at which to evaluate cdf(s)

    Returns
    -------
    cumulative distribution function for the unit gaussian at x

    '''
    return 0.5*(1. + tf.math.erf(x/np.sqrt(2)))


def gaussian_integral_on_unit_interval(mean, width, backend = K):
    '''

    Parameters
    ----------
    mean : mean(s) of unit gaussian(s)
    width : width(s) of unit gaussian(s)
    backend : TYPE, optional
        DESCRIPTION. The default is K.

    Returns
    -------
    integral of unit gaussian on [0,1]

    '''
    z0 = (0. - mean)/width
    z1 = (1. - mean)/width
    integral = cumulative(z1) - cumulative(z0)
    if backend == np:
        integral = K.eval(integral)
    return integral 


def gaussian(x, coeff, mean, width, backend = K):
    '''

    Parameters
    ----------
    x : variable value(s) at which to evaluate unit gaussian(s)
    coeff : normalization constant(s) for unit gaussian(s)
    mean : mean(s) of unit gaussian(s)
    width : width(s) of unit gaussian(s)

    Returns
    -------
    function value of unit gaussian(s) evaluated at x

    '''
    return coeff*backend.exp(-backend.square(x - mean)/(2.*backend.square(width)))/backend.sqrt(2.*backend.square(width)*np.pi)


def GMM(x, coeff, mean, width, num_gmm, backend = K):
    '''

    Parameters
    ----------
    x : variable value(s) at which to evaluate unit gaussian(s)
    coeff : normalization constant(s) for unit gaussian(s)
    mean : mean(s) of unit gaussian(s)
    width : width(s) of unit gaussian(s)
    num_gmm : TYPE
        DESCRIPTION.

    Returns
    -------
    posterior distribution function (GMM)

    '''
    pdf = backend.zeros_like(x)
    
    for i in range(num_gmm):
        cmp = gaussian(x, coeff[:,i], mean[:,i], width[:,i], backend = backend)
        cmp = cmp/gaussian_integral_on_unit_interval(mean[:,i], width[:,i], backend = backend)
        pdf += cmp
        
    return pdf