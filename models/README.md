# Model Classes

##  Actor Critic Models

in Actor Critic methods we have two predictions: the actor or policy predicts what actions
to take given the sate.  The critic predicts the value or expected discounted sum of rewards 
the actor will recieve from that action in that state.

These could be two completely different models or they could be two 
head networks branching off of one network.


### ActorCriticBase
memeber funcitons for all actor critic models


### SeperateActorCritic
this template alows the user to define seperate networks 
with variable convolutional layers followed by dense layers