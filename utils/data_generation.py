from random import randint,choice,seed

import numpy as np
import pandas as pd

from utils.cube_2x2 import movement

def dataGenerator(n_runs:int=50,n_moves:int=20,random_state:int=None):
    if random_state:
        seed(random_state)
    cube22 = np.array([
                      [[0,0],[0,0]], #front
                      [[1,1],[1,1]], #back
                      [[2,2],[2,2]], #upper
                      [[3,3],[3,3]], #botton
                      [[4,4],[4,4]], #right
                      [[5,5],[5,5]]  #left
                      ])

    
    
    
    positions = []
    moves = []
    runs = []
    move_numbers = []
    for j in range(n_runs):

        cube22 = np.array([
                          [[0,0],[0,0]], #front
                          [[1,1],[1,1]], #back
                          [[2,2],[2,2]], #upper
                          [[3,3],[3,3]], #botton
                          [[4,4],[4,4]], #right
                          [[5,5],[5,5]]  #left
                          ])
        
        positions.append(cube22.flatten())
        moves.append('x')  
        runs.append(j)
        move_numbers.append(0)        
        for i in range(n_moves):
            side = randint(0,5)
            ori = choice([0,2])
            cube22= movement(cube22,side,ori)
            
            if ori==0 and any(cube22.flatten() != positions[0]):
                moves.append(str(side)+','+'2')
            elif ori==2 and any(cube22.flatten() != positions[0]):
                moves.append(str(side)+','+'0')
            else:
                moves.append('x')
                
            positions.append(cube22.flatten())
            runs.append(j) 
            move_numbers.append(i+1)
        
   
    cubeDF = pd.DataFrame(data=positions,columns = range(24))
    cubeDF['Run'] = runs
    cubeDF['Move_number'] = move_numbers
    cubeDF['Move'] = moves
    return cubeDF
