import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import  numpy as np
import logging

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

"""
This is actully two models for actor critic
    self.logits: actor/ policy
    self.value: value  
"""

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')

        # actor policy network
        self.actor_hidden = kl.Dense(128, activation='relu')
        self.actor_logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

        # value network
        self.value_hidden = kl.Dense(128, activation='relu')
        self.value_out = kl.Dense(1, name='value')


    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs)
        # separate hidden layers from the same input tensor
        hidden_logs = self.actor_hidden(x)
        hidden_vals = self.value_hidden(x)
        return self.actor_logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
