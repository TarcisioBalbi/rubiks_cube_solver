import pickle
import copy

import numpy as np
import numpy as np
from random import randint,choice,seed

from utils.cube_2x2 import movement,cubeView,moveMap

def play(cube,model): 
    same_move_buffer = ['-1,-1']  # starting buffer with an invalid value
    while True: #simulanting a Do-While structure
        cubeView(cube)
        vector = cube.flatten()
        move = str(model.predict(vector.reshape(1,-1))[0])
        print(move)
        if move == 'x':
            return "solved", cube
        else:
            side,ori = moveMap(move)
            cube = movement(cube,side,ori)
            print(move[0],same_move_buffer)
            if (move[0] == same_move_buffer[-1][0]):
                same_move_buffer.append(move)
            else:
                same_move_buffer = ['-1,-1']

            if len(same_move_buffer)>5:
                return "Stuck",cube
            print(same_move_buffer)
    

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
print('Scrambling cube')
# seed(10)
for i in range(5):
    side = randint(0,5)
    print(side)
    ori = choice([0,2])
    cube22= movement(cube22,side,ori)
    cubeView(cube22)

print('Playing')
result,cube = play(cube22,rfc)
cubeView(cube,persist=True)
if all(cube.flatten() == reference_cube.flatten()):
    print("Sucess! Model result: ",result)
else:
    print('Fail! Model result: ',result)




