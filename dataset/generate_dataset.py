import numpy as np
import argparse

np.random.seed(9)

def genGoal(grid, connection_percent_th = 80):
    def dfs(x, y):
        visited[x, y] = True
        for action in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if not visited[x + action[0], y + action[1]]:
                dfs(x + action[0], y + action[1]) 
    
    visited = np.copy(grid)
    room_xy = np.where(grid == 0)

    for _ in range(10): # chance for connnection_percent ... should we reset visited each time?
        k = np.random.choice(room_xy[0].size) #### pick a random room as a start
        goal_r = room_xy[0][k]
        goal_c = room_xy[1][k]
        dfs(goal_r, goal_c)
        if np.mean(visited) > connection_percent_th / 100:
            break
    grid[np.where(visited == False)] = 1
    grid[goal_r, goal_c] = 2
    return grid

def genStart(grid):
    room_xy = np.where(grid == 0)
    k = np.random.choice(room_xy[0].size)
    start_r = room_xy[0][k]
    start_c = room_xy[1][k]
    grid[start_r, start_c] = 3
    return grid

def rcToRoomIndex(grid, r, c):
    if grid[r, c]:
        print ("This cell is not empty!")
        return -1
    room_rc = np.where(grid == 0) 
    this_row = np.where(room_rc[0] == r)[0]
    this_row_column = np.where(room_rc[1][this_row] == c)[0] ### there should be only one
    return this_row[0] + this_row_column[0] 


def roomIndexToRc(grid, room_index):
    room_rc = np.where(grid == 0) 
    return room_rc[0][room_index], room_rc[1][room_index] 


def genMaps(num_maps = 3, map_side_len = 28, generator='small', density = 4, reps=4):
    if generator == 'sparse':
        layout = np.random.randint(0, 100, size = (num_maps, map_side_len, map_side_len)) < density #room (empty pixel) -> 0, obstacles and wall -> 1
    if generator == 'small':
        layout = np.zeros([num_maps, map_side_len, map_side_len], dtype = np.uint) 
        for i in range(len(layout)): # generate obstacle_num of random obstacles for each map
            obs_x = np.random.randint(0, map_side_len, density)
            obs_y = np.random.randint(0, map_side_len, density)
            layout[i, obs_x, obs_y] = 1
    obstacle_maps = np.ones([num_maps, map_side_len + 2, map_side_len + 2], dtype = np.uint)
    obstacle_maps [:, 1 : -1 , 1 : -1] = layout
    maps = np.array([genGoal(grid) for grid in obstacle_maps])
    np.tile(maps, (reps, 1, 1))
    maps = [genStart(grid) for grid in maps]
    return np.array(maps)

def main(n_envs=1024, size=8, generator='sparse', density=20, reps=4, train=True):
    if train is True:
        pth = 'train'
    else:
        pth = 'test'
        reps = 1
    save_path = f'dataset/{pth}_worlds/{generator}_{size}_{density}_{n_envs}' # uhhh

    maps = genMaps(num_maps=n_envs, map_side_len=size,  generator=generator, density=density, reps=reps)

    np.save(save_path, maps)

# allow args to be passed in :D
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", "-ne", type=int, help="number of environments", default=2000)
    parser.add_argument("--size", "-s", type=int, help="side length of map", default=4)
    parser.add_argument("--generator", "-g", type=str, help="type of environment", default="small")
    parser.add_argument("--density", "-d", type=int, help="percent/num of obstacles", default=4)
    parser.add_argument("--reps", '-r', type=int, help="n times gridworld is repeated with diff. initial states", default=1)
    parser.add_argument("--train", '-t', type=bool, help="train or test", default=False)
    args = parser.parse_args()
    
    main(args.n_envs, args.size, args.generator, args.density, args.reps, args.train)