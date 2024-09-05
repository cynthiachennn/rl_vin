import sys
import numpy as np
import argparse
import pickle

sys.path.append('.')
from generators.map_generator import MapGenerator
sys.path.remove('.')

class SparseMap(MapGenerator):    
    @staticmethod
    def genMaps(num_maps = 3, map_side_len = 28, obstacle_percent = 5, scale = 3):  # Sami prefers keeping scale = 1 and make the obstacle_percentage very low
        small_map_side = int(map_side_len  / scale)
        layout = np.random.randint(0, 100, size = (num_maps, small_map_side, small_map_side)) < obstacle_percent  #room (empty pixel) -> 0, obstacles and wall -> 1   also we should always add borders
        large_layout = np.zeros([num_maps, map_side_len, map_side_len], dtype = np.uint) 
        for r in range(small_map_side):
            for c in range(small_map_side):
                large_layout[ : , r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = layout[:, r, c].reshape(-1, 1, 1) 
        obstacle_maps = np.ones([num_maps, map_side_len + 2, map_side_len + 2], dtype = np.uint)
        obstacle_maps [:, 1 : -1 , 1 : -1] = large_layout

        maps = [SparseMap.genGoal(grid, 80) for grid in obstacle_maps]
        return maps
    
class SmallMap(MapGenerator):
    @staticmethod
    def genMaps(num_maps = 3, map_side_len = 5, obstacle_num = 5):
        small_maps = np.zeros([num_maps, map_side_len, map_side_len], dtype = np.uint) 
        for i in range(len(small_maps)): # generate obstacle_num of random obstacles for each map
            obs_x = np.random.randint(0, map_side_len, obstacle_num)
            obs_y = np.random.randint(0, map_side_len, obstacle_num)
            small_maps[i, obs_x, obs_y] = 1
        obstacle_maps = np.ones([num_maps, map_side_len + 2, map_side_len + 2], dtype = np.uint)
        obstacle_maps [:, 1 : -1 , 1 : -1] = small_maps
        maps = [SmallMap.genGoal(grid, 80) for grid in obstacle_maps]
        return maps
    
    @staticmethod
    def genGoal(grid, connection_percent_th = 80):
        def dfs(x, y):
            visited[x, y] = True
            for action in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if not visited[x + action[0], y + action[1]]:
                    dfs(x + action[0], y + action[1]) 
        
        visited = np.copy(grid)
        room_xy = np.where(grid == 0) 
        for _ in range(10):
            goal_r = np.random.randint(1, grid.shape[0] - 1, 1)[0]
            goal_c = np.random.randint(1, grid.shape[1] - 1, 1)[0]
            if grid[goal_r, goal_c]:
                continue
            dfs(goal_r, goal_c)
            if np.mean(visited) > connection_percent_th / 100:
                break
        grid[np.where(visited == 0)] = 1
        grid[goal_r, goal_c] = 2

        return grid

def main(n_envs=1024, size=8, density=20, scale=2, type='sparse'):
    save_path = f'dataset/train_worlds/{type}_{size}_{density}_{n_envs}' # uhhh

    if type == 'sparse':
        maps = SparseMap.genMaps(num_maps=n_envs, map_side_len=size, obstacle_percent=density, scale=scale)
    if type == 'small':
        maps = SmallMap.genMaps(num_maps=n_envs, map_side_len=size, obstacle_num=density)

    maps = np.array(maps)

    np.save(save_path, maps)

    
# allow args to be passed in :D
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", "-ne", type=int, help="number of environments", default=10000)
    parser.add_argument("--size", "-s", type=int, help="side length of map", default=6)
    parser.add_argument("--density", "-d", type=int, help="percent/num of obstacles", default=6)
    parser.add_argument("--scale", "-sc", type=int, help="scaling factor", default=1)
    parser.add_argument("--type", "-t", type=str, help="type of environment", default="small")
    args = parser.parse_args()
    
    main(args.n_envs, args.size, args.density, args.scale, args.type)

