from asyncio import start_server
from tracemalloc import start
import numpy as np
# this file and generate_dataset.py are somewhat redundant... should probably seperate out the classes in generate_dataset from the generation file...

class MapGenerator():
    @staticmethod
    def genGoal(map, connection_percent_th = 80):
        def dfs(x, y):
            visited[x, y] = True
            for action in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if not visited[x + action[0], y + action[1]]:
                    dfs(x + action[0], y + action[1]) 
        
        visited = np.copy(map)
        room_xy = np.where(map == 0)

        for _ in range(10): # chance for connnection_percent ... should we reset visited each time?
            k = np.random.choice(room_xy[0].size) #### pick a random room as a start
            goal_r = room_xy[0][k]
            goal_c = room_xy[1][k]
            dfs(goal_r, goal_c)
            if np.mean(visited) > connection_percent_th / 100:
                break
        map[np.where(visited == False)] = 1
        map[goal_r, goal_c] = 2

        return map

    @staticmethod
    def genStart(map):
        room_xy = np.where(map == 0)
        k = np.random.choice(room_xy[0].size)
        start_r = room_xy[0][k]
        start_c = room_xy[1][k]
        
        map[start_r, start_c] = 3

        return map
    
    @staticmethod 
    def rcToRoomIndex(grid, r, c):
        if grid[r, c]:
            print ("This cell is not empty!")
            return -1
        room_rc = np.where(grid == 0) 
        this_row = np.where(room_rc[0] == r)[0]
        this_row_column = np.where(room_rc[1][this_row] == c)[0] ### there should be only one
        return this_row[0] + this_row_column[0] 


    @staticmethod 
    def roomIndexToRc(grid, room_index):
        room_rc = np.where(grid == 0) 
        return room_rc[0][room_index], room_rc[1][room_index] 

   


