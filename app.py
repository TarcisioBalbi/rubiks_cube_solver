import pickle
import copy
from random import seed

import numpy as np
import numpy as np
import cv2


from utils.cube_2x2 import movement,cubeView,moveMap,scramble_cube

def play(cube,model,n_scramble_moves:int=3,random_state:int=None):
    if random_state:
        seed(random_state)
    cube = scramble_cube(cube,n_scramble_moves) 
    cv2.startWindowThread()
    cv2.namedWindow('Game')
    cv2.moveWindow('Game', 40,30) 
    same_move_buffer = ['-1,-1']  # starting buffer with an invalid value
    while True: #simulanting a Do-While structure
        cubeView(cube,window_name='Game')
        vector = cube.flatten()
        move = str(model.predict(vector.reshape(1,-1))[0])

        if move == 'x':
            cubeView(cube,window_name='Game',persist=True)
            return "solved", cube
        else:
            side,ori = moveMap(move)
            cube = movement(cube,side,ori)

            if (move[0] == same_move_buffer[-1][0]):
                same_move_buffer.append(move)
            else:
                same_move_buffer = ['-1,-1']

            if len(same_move_buffer)>5:
                return "Stuck",cube

fileName = 'data/model.sav'

with open(fileName,'rb') as file:
    rfc = pickle.load(file)

letsPlay = True
reference_cube = np.array([
                    [[0,0],[0,0]], #front
                    [[1,1],[1,1]], #back
                    [[2,2],[2,2]], #upper
                    [[3,3],[3,3]], #botton
                    [[4,4],[4,4]], #right
                    [[5,5],[5,5]]  #left
                    ])

cube22 = copy.deepcopy(reference_cube)

print('Playing')
result,cube = play(cube22,rfc,n_scramble_moves=7,random_state=10)

if all(cube.flatten() == reference_cube.flatten()):
    print("Sucess! Model result: ",result)
else:
    print('Fail! Model result: ',result)




