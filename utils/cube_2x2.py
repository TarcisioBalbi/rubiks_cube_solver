from random import randint,choice

import numpy as np
import matplotlib.pyplot as plt
import cv2

white = (255,255,255)
yellow = (255,255,0)
blue = (0,0,255)
green = (0,255,0)
red = (255,0,0)
orange = (255,128,0)

def colorCube(cube):
    cCube = np.empty((6,2,2,3),np.uint8)
    
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            for k in range(cube.shape[2]):
                #print(cube[i,j,k])
                if cube[i,j,k]== 0:
                    cCube[i,j,k] = white
                    
                elif cube[i,j,k] == 1:
                    cCube[i,j,k] = yellow
                    
                elif cube[i,j,k] == 2:
                    cCube[i,j,k] = green
                    
                elif cube[i,j,k] == 3:
                    cCube[i,j,k] = blue
                    
                elif cube[i,j,k] == 4:
                    cCube[i,j,k] = orange
                    
                    
                elif cube[i,j,k] == 5:
                    cCube[i,j,k] = red
                    
                else:
                    print('Invalid Color!')
   
    return cCube


def cubeView(cube,persist:bool=False,window_name:str='Game'):
    
    view = np.zeros((3,8,6),np.uint8)
    c = colorCube(cube)
   
    view[:,2,0] = c[2,0,0]
    view[:,3,0] = c[2,0,1]
    view[:,2,1] = c[2,1,0]
    view[:,3,1] = c[2,1,1]

    view[:,2,2] = c[0,0,0]
    view[:,3,2] = c[0,0,1]
    view[:,2,3] = c[0,1,0]
    view[:,3,3] = c[0,1,1]

    view[:,2,4] = c[3,0,0]
    view[:,3,4] = c[3,0,1]
    view[:,2,5] = c[3,1,0]
    view[:,3,5] = c[3,1,1]

    view[:,0,2] = c[5,0,0]
    view[:,1,2] = c[5,0,1]
    view[:,0,3] = c[5,1,0]
    view[:,1,3] = c[5,1,1]

    view[:,4,2] = c[4,0,0]
    view[:,5,2] = c[4,0,1]
    view[:,4,3] = c[4,1,0]
    view[:,5,3] = c[4,1,1]
    
    view[:,6,2] = c[1,0,0]
    view[:,7,2] = c[1,0,1]
    view[:,6,3] = c[1,1,0]
    view[:,7,3] = c[1,1,1]

    res = cv2.resize(np.swapaxes(view,2,0,), dsize=(600, 400), interpolation=cv2.INTER_NEAREST )
    cv2.imshow(window_name,res)
    if persist:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.waitKey(500) 
       
def sidesMap(side):
    if side == 0:
        return 1,2,3,4,5
    elif side == 1:
        return 0,2,3,5,4
    elif side == 2:
        return 3,1,0,4,5
    elif side == 3:
        return 2,0,1,4,5
    elif side == 4:
        return 5,2,3,1,0
    elif side == 5:
        return 4,2,3,0,1
    else:
        print('Invalid Side!')
   


def moveMap(movement):
    return int(movement[0]),int(movement[2])

def sidesMap(side):
    if side == 0:
        return 1,2,3,4,5
    elif side == 1:
        return 0,2,3,5,4
    elif side == 2:
        return 3,1,0,4,5
    elif side == 3:
        return 2,0,1,4,5
    elif side == 4:
        return 5,2,3,1,0
    elif side == 5:
        return 4,2,3,0,1
    else:
        print('Invalid Side!')

def movement(cube,side,direction):
    #sides: 0 = front, 1= back, 2 = upper, 3 = botton, 4 = right, 5 = left
    #directions: 0 = 90° clockwise , 2 = 90° anti-clockwise
    direction -=1
    cube[side] = np.rot90(cube[side],direction)
    
    oposite,upper,botton,right,left = sidesMap(side)

    if side==0:
        if direction == -1:
    
            temp1,temp2 = cube[2,1,0],cube[2,1,1]
            
            cube[2,1,0], cube[2,1,1] = cube[5,0,1], cube[5,1,1]
            
            cube[5,0,1], cube[5,1,1] = cube[3,0,0],cube[3,0,1]
            
            cube[3,0,0],cube[3,0,1] = cube[4,0,0], cube[4,1,0]
            
            cube[4,0,0], cube[4,1,0] = temp1,temp2
            
        elif direction == 1:
            
            temp1,temp2 = cube[2,1,0],cube[2,1,1]
            
            cube[2,1,0], cube[2,1,1] = cube[4,0,0], cube[4,1,0]
            
            cube[4,0,0], cube[4,1,0] = cube[3,0,0],cube[3,0,1]
            
            cube[3,0,0],cube[3,0,1] = cube[5,0,1], cube[5,1,1]
            
            cube[5,0,1], cube[5,1,1] = temp1,temp2
            
    elif side==1:
        if direction == -1:
    
            temp1,temp2 = cube[2,0,0],cube[2,0,1]
            
            cube[2,0,0], cube[2,0,1] = cube[4,0,1], cube[4,1,1]
            
            cube[4,0,1], cube[4,1,1] = cube[3,1,1],cube[3,1,0]
            
            cube[3,1,1],cube[3,1,0] = cube[5,0,0], cube[5,1,0]
            
            cube[5,0,0], cube[5,1,0] = temp1,temp2
            
        elif direction == 1:
            
            temp1,temp2 = cube[2,0,0],cube[2,0,1]
            
            cube[2,0,0], cube[2,0,1] = cube[5,0,0], cube[5,1,0]
            
            cube[5,0,0], cube[5,1,0] = cube[3,1,1],cube[3,1,0]
            
            cube[3,1,1],cube[3,1,0] = cube[4,0,1], cube[4,1,1] 
            
            cube[4,0,1], cube[4,1,1] = temp1,temp2
            
    elif side==2:
        if direction == -1:
    
            temp1,temp2 = cube[1,0,0],cube[1,0,1]
            
            cube[1,0,0], cube[1,0,1] = cube[5,0,0], cube[5,0,1]
            
            cube[5,0,0], cube[5,0,1] = cube[0,0,0],cube[0,0,1]
            
            cube[0,0,0],cube[0,0,1] = cube[4,0,0], cube[4,0,1]
            
            cube[4,0,0], cube[4,0,1] = temp1,temp2
            
        elif direction == 1:
            
            temp1,temp2 = cube[1,0,0],cube[1,0,1]
            
            cube[1,0,0], cube[1,0,1] = cube[4,0,0], cube[4,0,1]
            
            cube[4,0,0], cube[4,0,1] = cube[0,0,0],cube[0,0,1]
            
            cube[0,0,0], cube[0,0,1] = cube[5,0,0], cube[5,0,1] 
            
            cube[5,0,0], cube[5,0,1] = temp1,temp2
            
    elif side==3:
        if direction == -1:
    
            temp1,temp2 = cube[1,1,0],cube[1,1,1]
            
            cube[1,1,0], cube[1,1,1] = cube[4,1,0], cube[4,1,1]
            
            cube[4,1,0], cube[4,1,1] = cube[0,1,0],cube[0,1,1]
            
            cube[0,1,0],cube[0,1,1] = cube[5,1,0], cube[5,1,1]
            
            cube[5,1,0], cube[5,1,1] = temp1,temp2
            
        elif direction == 1:
            
            temp1,temp2 = cube[1,1,0],cube[1,1,1]
            
            cube[1,1,0], cube[1,1,1] = cube[5,1,0], cube[5,1,1]
            
            cube[5,1,0], cube[5,1,1]  = cube[0,1,0],cube[0,1,1]
            
            cube[0,1,0],cube[0,1,1] =  cube[4,1,0], cube[4,1,1]
            
            cube[4,1,0], cube[4,1,1] = temp1,temp2
            
    elif side==4:
        if direction == -1:
    
            temp1,temp2 = cube[2,0,1],cube[2,1,1]
            
            cube[2,0,1],cube[2,1,1] = cube[0,0,1], cube[0,1,1]
            
            cube[0,0,1], cube[0,1,1] = cube[3,0,1],cube[3,1,1]
            
            cube[3,0,1],cube[3,1,1] = cube[1,1,0], cube[1,0,0]
            
            cube[1,1,0], cube[1,0,0] = temp1,temp2
            
        elif direction == 1:
            
            temp1,temp2 = cube[2,0,1],cube[2,1,1]
            
            cube[2,0,1],cube[2,1,1] =  cube[1,1,0], cube[1,0,0]
            
            cube[1,1,0], cube[1,0,0] = cube[3,0,1],cube[3,1,1]
            
            cube[3,0,1],cube[3,1,1] =  cube[0,0,1], cube[0,1,1]
            
            cube[0,0,1], cube[0,1,1] = temp1,temp2
    elif side==5:
        if direction == -1:
    
            temp1,temp2 = cube[2,0,0],cube[2,1,0]
            
            cube[2,0,0],cube[2,1,0] = cube[1,1,1], cube[1,0,1]
            
            cube[1,1,1], cube[1,0,1] = cube[3,0,0],cube[3,1,0]
            
            cube[3,0,0],cube[3,1,0] = cube[0,0,0], cube[0,1,0]
            
            cube[0,0,0], cube[0,1,0] = temp1,temp2
            
        elif direction == 1:
            
            temp1,temp2 = cube[2,0,0],cube[2,1,0]
            
            cube[2,0,0],cube[2,1,0] =  cube[0,0,0], cube[0,1,0]
            
            cube[0,0,0], cube[0,1,0] = cube[3,0,0],cube[3,1,0]
            
            cube[3,0,0],cube[3,1,0] = cube[1,1,1], cube[1,0,1]
            
            cube[1,1,1], cube[1,0,1]  = temp1,temp2
            
            
       
    return cube

def scramble_cube(cube,n_moves:int=5):
    print('Scrambling cube')
    for i in range(n_moves):
        side = randint(0,5)
        ori = choice([0,2])
        cube= movement(cube,side,ori)
    return cube