import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import logging

class ValueLoss:
    def __init__(self,
                 value_coeff=0.05):
        self.value_coeff = value_coeff

    def get_loss(self,
                 returns,
                 value):
        # value loss is typically MSE between value estimates and returns
        return self.value_coeff * kls.mean_squared_error(returns, value)