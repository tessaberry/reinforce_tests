import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.initializers as kinit
import numpy as np
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

class SeparateActorCritic(tf.keras.Model):
    def __init__(self,
                 num_actions,
                 actor_conv_layers=[(32, 8, 4), (64, 4, 2)],
                 actor_dense_layers=[2],
                 critic_conv_layers=[(32, 8, 4), (64, 4, 2)],
                 critic_dense_layers=[2]
                 ):
        super().__init__('mlp_policy')
        self.actor_conv_layers = actor_conv_layers
        self.actor_dense_layers = actor_dense_layers
        self.critic_conv_layers = critic_conv_layers
        self.critic_dense_layers = critic_dense_layers

        # actor policy network

        if self.actor_conv_layers is not None:
            self.actor_conv_net = [
                kl.Conv2D(layer[0],
                        layer[1],
                        strides=layer[2],
                        activation='relu',
                        kernel_initializer=kinit.Orthogonal(np.sqrt(2)),
                        name='Conv' + str(i))
                    for i, layer in enumerate(self.actor_conv_layers)]
        self.actor_dense_net = [
                kl.Dense(layer,
                      activation='relu',
                      name='Dense' + str(i),
                      kernel_initializer=kinit.Orthogonal(np.sqrt(2)))
                  for i, layer in enumerate(self.actor_dense_layers)]

        self.actor_logits = kl.Dense(num_actions,
                                     activation='linear',
                                     kernel_initializer=kinit.Orthogonal(0.01),
                                     name='policy_logits')
        self.dist = ProbabilityDistribution()

        # value network
        if self.critic_conv_layers is not None:
            self.value_conv_net = [
                kl.Conv2D(layer[0],
                        layer[1],
                        strides=layer[2],
                        activation='relu',
                        kernel_initializer=kinit.Orthogonal(np.sqrt(2)),
                        name='Conv' + str(i))
                    for i, layer in enumerate(self.critic_conv_layers)]
        self.value_dense_net = [
                kl.Dense(layer,
                      activation='relu',
                      name='Dense' + str(i),
                      kernel_initializer=kinit.Orthogonal(np.sqrt(2)))
                  for i, layer in enumerate(self.critic_dense_layers)]

        self.value_out = kl.Dense(1,
                                  activation='linear',
                                  kernel_initializer=kinit.Orthogonal(0.01),
                                  name='value')


    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        xa = tf.convert_to_tensor(inputs)
        xv = tf.convert_to_tensor(inputs)

        # run the actor model
        if self.actor_conv_layers is not None:
            for layer in self.actor_conv_net:
                xa = layer(xa)

        for layer in self.actor_dense_net:
            xa = layer(xa)

        actor_logits = self.actor_logits(xa)

        # run the value model
        if self.critic_conv_layers is not None:
            for layer in self.critic_conv_net:
                xv = layer(xv)
        for layer in self.value_dense_net:
            xv = layer(xv)
        value_out = self.value_out(xv)

        return actor_logits, value_out

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

    def get_logits(self):
        logits, value = self.predict(obs)
        return logits
