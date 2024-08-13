from generators.map_generator import MapGenerator
from generators.twoD_map_path import TwoDMapPath
import numpy as np
import pickle 

import sys
sys.path.append('..')

class SparseMap(MapGenerator):    
    @staticmethod
    def genMaps(num_maps = 3, map_side_len = 28, obstacle_percent = 5, scale = 3 , solvers = ['QMDP']):   # Sami prefers keeping scale = 1 and make the obstacle_percentage very low
        small_map_side = int(map_side_len  / scale)
        layout = np.random.randint(0, 100, size = (num_maps, small_map_side, small_map_side)) < obstacle_percent  #freespace -> 0, obstacles and wall -> 1   also we should always add borders
        large_layout = np.zeros([num_maps, map_side_len, map_side_len], dtype = np.uint) 
        for r in range(small_map_side):
            for c in range(small_map_side):
                large_layout[ : , r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = layout[:, r, c].reshape(-1, 1, 1) 
        obstacle_maps = np.ones([num_maps, map_side_len + 2, map_side_len + 2], dtype = np.uint)
        obstacle_maps [:, 1 : -1 , 1 : -1] = large_layout

        all_2d_maps = []
        for m in range(num_maps):
            goal_r, goal_c = SparseMap.genGoal(obstacle_maps[m], 80)
            all_2d_maps.append(TwoDMapPath(obstacle_maps[m], goal_r, goal_c, solvers))    
            
        return all_2d_maps