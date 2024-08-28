import numpy as np


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
        for _ in range(10): ### we have 10 chance for connection_percent_th  
            #k = np.random.choice(room_xy[0].size) #### peak a random room as a goal
            # goal_r = np.random.randint(map.shape[0] // 2 - 3, map.shape[0] // 2 + 3, 1)[0] # room_xy[0][k]
            # goal_c = np.random.randint(map.shape[1] // 2 - 3, map.shape[1] // 2 + 3, 1)[0] #room_xy[1][k]
            goal_r = np.random.randint(0, map.shape[0] // 2 + 3, 1)[0] # room_xy[0][k]
            goal_c = np.random.randint(0, map.shape[1] // 2 + 3, 1)[0] #room_xy[1][k]
            if map[goal_r, goal_c]:
                continue
            dfs(goal_r, goal_c)
            if np.mean(visited) > connection_percent_th / 100:
                break
                
        map[np.where(visited == 0)] = 1 ## make the rest (disconnected rooms), obstacle
        return goal_r, goal_c

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

   


