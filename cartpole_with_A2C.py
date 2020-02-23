import gym
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

#from models.simple_dense_model_v0 import Model
from models.seperate_actor_critic import SeparateActorCritic
from agents.A2C_agent import A2CAgent

##-----------------
##  Set Up
##-----------------
# set up a logger
logging.getLogger().setLevel(logging.INFO)

# set up the game environment
# here we are using openAI gym's cartpole
env = gym.make('CartPole-v0')
model = SeparateActorCritic(num_actions=env.action_space.n,
                            actor_conv_layers=None,
                            critic_conv_layers=None)
agent = A2CAgent(model)

##-----------------
##  Train
##-----------------

rewards_history = agent.train(env, batch_sz=50, updates=100)
print("Finished training.")
print("Total Episode Reward: %d out of 200" % agent.test(env, True))

##-----------------
##  Vizualize
##-----------------
plt.style.use('seaborn')
plt.plot(np.arange(0, len(rewards_history), 25), rewards_history[::25])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()