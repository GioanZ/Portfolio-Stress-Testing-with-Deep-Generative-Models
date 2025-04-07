"""
Copyright:
    Portfolio Stress Testing with Deep Generative Models
    github.com/GioanZ
Disclaimer:
    This software is for research and educational purposes only.
    It is NOT intended for actual financial decision-making or investment strategies.
    The authors assume no liability for any losses or damages arising from the use
    of this code. Users should conduct their own due diligence before making financial
    decisions.

    This project utilizes deep generative models to simulate financial stress testing.
    The models are trained on historical market and macroeconomic data, but all results
    should be interpreted with caution.
"""

import keras.saving

import tensorflow as tf
from tensorflow.keras.losses import mse
from tensorflow.keras import layers, backend as K

import tensorflow_probability as tfp

SEED_RANDOM = 29
tf.random.set_seed(SEED_RANDOM)

""" Define custom layers: Sampling, Reconstruction Loss, KL Divergence """


@keras.saving.register_keras_serializable()
class SamplingLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var + 1e-8) * epsilon


@keras.saving.register_keras_serializable()
class ReconstructionLossLayer(layers.Layer):
    def __init__(self, weight_tail=1.0, **kwargs):
        super(ReconstructionLossLayer, self).__init__(**kwargs)
        self.weight_tail = weight_tail

    def call(self, inputs):
        x, x_pred = inputs
        # Standard MSE loss
        mse_loss = K.mean(mse(x, x_pred))
        # Compute the 5th percentile for x and x_pred along the batch axis
        p_true = tfp.stats.percentile(x, 5.0, interpolation="linear", axis=0)
        p_pred = tfp.stats.percentile(x_pred, 5.0, interpolation="linear", axis=0)
        # To obtain a single value, take the mean across all assets
        tail_loss = tf.reduce_mean(tf.abs(p_true - p_pred))
        # Sum the loss, giving more or less weight to the tail term as needed
        loss = mse_loss + self.weight_tail * tail_loss
        self.add_loss(loss)
        return x_pred


@keras.saving.register_keras_serializable()
class KLDivergenceLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
        )
        self.add_loss(tf.reduce_mean(kl_loss))
        return z_mean
