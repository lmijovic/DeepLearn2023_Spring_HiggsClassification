# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:12:01 2021

@author: keira
"""

import keras.backend as K
from keras.layers import Layer
import ops
import tensorflow as tf
import tensorflow.compat.v1 as v1

tf.compat.v1.disable_eager_execution()
print('Eager exc', tf.executing_eagerly())
print('tensorflow: %s' % tf.__version__)


num_gradient_reversals = 0

def ReverseGradient (hp_lambda):
    """
    Function factory for gradient reversal, implemented in TensorFlow.
    """

    def reverse_gradient_function (X, hp_lambda=hp_lambda):
        """Flips the sign of the incoming gradient during training."""
        global num_gradient_reversals
        grad_name = "GradientReversal{}".format(num_gradient_reversals)
        num_gradient_reversals += 1
        @tf.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * hp_lambda]

        #g = K.get_session().graph
        #g = tf.Graph()
        g = v1.keras.backend.get_session().graph
        with g.gradient_override_map({'Identity': grad_name}):
            y = tf.identity(X)

        return y

    return reverse_gradient_function


class GradientReversalLayer(Layer):
    
    def __init__(self, hp_lambda, **kwargs):
        
        super(GradientReversalLayer, self).__init__(**kwargs)
        
        self.supports_masking = False
        self.hp_lambda = hp_lambda
        self.gr_op = ReverseGradient(self.hp_lambda)
        
    def call(self, x, mask = None):
        '''
        

        Parameters
        ----------
        x : of the format [coeffs, means, widths, m]

        '''
        return self.gr_op(x)
    

class PosteriorLayer(Layer):
    
    def __init__(self, num_gmm, **kwargs):
        '''
        

        Parameters
        ----------
        nb_gmm : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        Custom layer, models the posterior probability distribution for the diphoton mass using
        a Gaussian mixture model (GMM)

        '''
        # Base class constructor
        super(PosteriorLayer, self).__init__(**kwargs)
        
        self.num_gmm = num_gmm
        
    
    def call(self, x, mask = None):
        '''
        

        Parameters
        ----------
        x : of the format [coeffs, means, widths, m] 
        mask : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        Main call method of the layer

        '''
        coeffs, means, widths, m = x
        
        pdf = ops.GMM(m[:,0], coeffs, means, widths, self.num_gmm)
        
        return K.flatten(pdf)