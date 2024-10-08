# RL implementation for VIN
RL implementation of the Value Iteration Network architecture described in 
[Value Iteration Networks](https://arxiv.org/abs/1602.02867), based on the
[pytorch implementation](https://github.com/kentsommer/pytorch-value-iteration-networks/tree/master?tab=readme-ov-file) 
by [@kentsommer](https://github.com/kentsommer/). Our personal interpretations of dataset generation and training regimine were made.

# Requirements
Exact requirements are listed in the `requirements.txt` file. `environment.yml` has also been included to list the directly installed dependencies.

# Generate the dataset

To generate the train and test dataset, run `dataset/generate_dataset.py`. The arguments are explained as follows
- n_envs: The numbers of maps generated for a file
- size: The length of one side of the map, all maps generated are squares, and include a border layer on each size so the final matrix size of the map will be size + 2
- type: There are two types of maps: "small" and "sparse" maps. "Small" maps will generate a fixed amount of obstacles, while "sparse" maps will generate a percentage of obstacles. 
- density: The amount of obstacles in a map. When "type" is "small", density refers to the amount of obstacles that will be placed. When "type" is "sparse", density refers to the chance any block will be an obstacle.
- reps: The number of times a single map layout is generated with a random starting position. The total amount of environments generated by the script is n_envs x reps
- train: Defaults to true, must be set to false to indicate that you are generating a test set so it can be put into the correct folder.

Sample datasets that were generated for the training and testing of this model are:

Train sets: 4x4, 8x8, 16x16

`python dataset/generate_dataset.py -ne 20000 -s 4 -g 'small' -d 4 -r 4 -t True`

`python dataset/generate_dataset.py -ne 20000 -s 8 -g 'sparse' -d 20 -r 4 -t True`

`python dataset/generate_dataset.py -ne 20000 -s 16 -g 'sparse' -d 20 -r 4 -t True`


Test sets: 4x4, 8x8, 16x16

`python dataset/generate_dataset.py -ne 2000 -s 4 -g 'small' -d 4 -r 1 -t False`

`python dataset/generate_dataset.py -ne 2000 -s 8 -g 'sparse' -d 20 -r 1 -t False`

`python dataset/generate_dataset.py -ne 2000 -s 16 -g 'sparse' -d 20 -r 1 -t False`


# Train the model
The model can be trained two ways - with imitation learning and reinforcement learning. Additionally, we used curriculum learning to train our RL agents, which refers to training our agent on a smaller state space before moving to a bigger one. The '--trainfiles' argument can take a list of training datasets, while '--testfile' specifies which final dataset to test the model on. The method of training can be switched from reinforcement learning and imitation learning by setting '--mode' to either 'rl' or 'expert'.

The following will use RL to train a model on the smallest 4x4 world dataset.

`python train.py -train dataset/train_worlds/small_4_4_20000.npy -test dataset/test_worlds/small_4_4_2000.npy` 


This uses RL with curriculum learning to learn from a 4x4 world to a 16x16 world.

`python train.py -train dataset/train_worlds/small_4_4_20000.npy dataset/train_worlds/sparse_16_20_20000.npy -test dataset/test_worlds/small_4_4_2000.npy` 

