# RL implementation for VIN
RL implementation of the Value Iteration Network architecture described in 
[Value Iteration Networks](https://arxiv.org/abs/1602.02867), based on the
[pytorch implementation](https://github.com/kentsommer/pytorch-value-iteration-networks/tree/master?tab=readme-ov-file) 
by [@kentsommer](https://github.com/kentsommer/)

Adapting this implementation to work with [koosha66](https://github.com/koosha66/)'s code and the generalized
POMDP format.

# To Do:
- finish gridworld generation script
- are there any ways to make this faster still ?
- clean up code a bit:
    - add way to pass arguments

- curricula learning
- reversal in this? like rotated map etc idkkk :\