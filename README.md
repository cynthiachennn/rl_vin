# RL implementation for VIN
RL implementation of the Value Iteration Network architecture described in 
[Value Iteration Networks](https://arxiv.org/abs/1602.02867), based on the
[pytorch implementation](https://github.com/kentsommer/pytorch-value-iteration-networks/tree/master?tab=readme-ov-file) 
by [@kentsommer](https://github.com/kentsommer/)

Files in `/dataset`, `domains`, `generators`, `utility` as well as `train.py` and `test.py` are from [the original implementation](https://github.com/kentsommer/) and mostly unmodified. `sandbox.py` contains the main current/in progress code that I am testing, and `sandbox_utils.py` contains the functional/completed code that makes up the script. Right now, `sandbox.py` generates a series of gridworlds, has the agent walk through the gridworlds in order to gain "experiences" and then attempts to learn the q value mapping of that grid world using a neural net that minimizes loss between the expected/calculated q values from the experiences and the predicted q values. 