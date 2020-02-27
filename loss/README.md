# Loss Classes

## Entropy Loss
encouraging some exploration through entropy can be a good thing in RL especially while
first learning a policy

https://fosterelli.co/entropy-loss-for-reinforcement-learning

Also on my todo list is to watch the Entropy on the tensorboard
as this can give clues about whether a learner has gotten stuck in a rut.

## PPO Loss

https://medium.com/@jonathan_hui/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12

We want to limit the step size that we can take in each step of gradient descent. 
We want to ensure that we are not changing our policy too drastically. We will
weight our trust region by the ratio of new and old policies

KL-divergence measures the difference between two data distributions p and q

`KLDiv = expectation(log(P/Q))`

We want the expectation of `L = pi_new/pi_old * advantages`

`L` is the expected advantage funtion for the new policy

`A = E[R] - Vbaseline`

We use the advantage function instead of the expected reward because it reduces the variance of the estimation

We want to make sure that the ratio of new and old policies don't 
move too much past an epsilon so we clip the KL divergence
