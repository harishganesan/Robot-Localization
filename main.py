import numpy as np
import matplotlib.pyplot as mp
import math
import scipy.stats as sc

range1 = np.loadtxt('ranges1.dat')
range2 = np.loadtxt('ranges2.dat')


bel = np.zeros((25,25,20))
bel2 = np.zeros((25,25,20))
bel_bar = np.zeros((25,25,20))
omega = 2 # v/r = 1/0.5 = 2
vel = 1
del_t = 0.2
theta_actual = np.zeros(bel.shape[2])

final_x1 = np.zeros(6)
final_y1 = np.zeros(6)
final_x2 = np.zeros(6)
final_y2 = np.zeros(6)
final_theta1 = np.zeros(6)
final_theta2 = np.zeros(6)
## Assuming the wall is the top row of the grid

for l in range(bel.shape[2]):
    theta_actual[l] = 0.165*l  # discretizing theta into 20 divisions of pi/19 in each slice
       
        
for i in range(bel.shape[0]):
    for j in range(bel.shape[1]):
        for k in range(bel.shape[2]):
            if(j == 0):   # Setting the y dimension to be 0 - we get only one slice of the 3d matrix
                bel[i][j][k] = 1/(bel.shape[0]*bel.shape[2])  # uniform initial probability
            else:
                bel[i][j][k] = 0
                
for i in range(bel2.shape[0]):
    for j in range(bel2.shape[1]):
        for k in range(bel2.shape[2]):
            if(j == 0):   # Setting the y dimension to be 0 - we get only one slice of the 3d matrix
                bel2[i][j][k] = 1/(bel2.shape[0]*bel2.shape[2])  # uniform initial probability
            else:
                bel2[i][j][k] = 0                


def beam(measurement,range1):
    obs = sc.multivariate_normal.pdf(measurement,mean=range1,cov=0.25)
    return obs


def observe1(measurement,x,y,theta):
    dist = y/np.sin(theta)   # Assuming wall is on the top
    obs = beam(measurement,dist)
    return obs



def observe2(measurement,x,y,theta):
    dist = y/np.sin(theta)   # Assuming wall is on the top
    obs1 = beam(measurement,dist)
    o = np.random.uniform(0,measurement,1) # sampling from 0 to measurement in a uniform way
    
    obs2 = beam(o,dist)
    #print(obs2)
    mul = obs1*0.8 + obs2*0.2    # uniform distribution
    return mul




def motion(x,y,theta):
        x_dash = x - (vel*np.sin(theta)/omega) + (vel*np.sin(theta + omega*del_t)/omega)
        y_dash = y + (vel*np.cos(theta)/omega) - (vel*np.cos(theta + omega*del_t)/omega)
        theta_dash = theta + omega * del_t
        return x_dash,y_dash, theta_dash

    
    

def bayes_filter1(measurement,bel):
    
    temp_bel = 0
    for i in range(bel.shape[0]):
        for j in range(bel.shape[1]):
            for k in range(bel.shape[2]):
        
                x_mot, y_mot, theta_mot = motion(i,j,theta_actual[k])
                  
                x_i = int(x_mot)
                y_i = int(y_mot)
                
                theta_d = np.full((bel.shape[2]),theta_mot)
                s3 = np.subtract(theta_actual , theta_d)
                s3 = np.absolute(s3)
                theta_index = np.argmin(s3)  # finds index of theta that's closest to theta from motion model 
                    
                
                bel[x_i][y_i][theta_index] += bel[i][j][k]
                
                bel_bar[i][j][k] =  temp_bel + bel[x_i][y_i][theta_index]*bel[i][j][k]
                temp_bel += bel[x_i][y_i][theta_index]*bel[i][j][k]

                measure = observe1(measurement,x_i,y_i,theta_mot)

                bel[i][j][k] = measure * bel_bar[i][j][k]
    bel_sum = 0
                
    for q in range(bel.shape[0]):
        for w in range(bel.shape[1]):
            for e in range(bel.shape[2]):
                bel_sum += bel[q][w][e]  # normalizing belief
    bel = bel/bel_sum
        
                        
    return bel


def bayes_filter2(measurement,bel):
    
    temp_bel = 0
    for i in range(bel.shape[0]):
        for j in range(bel.shape[1]):
            for k in range(bel.shape[2]):
        
                x_mot, y_mot, theta_mot = motion(i,j,theta_actual[k])
                
                x_i = int(x_mot)
                y_i = int(y_mot)
                
                theta_d = np.full((bel.shape[2]),theta_mot)
                s3 = np.subtract(theta_actual , theta_d)
                s3 = np.absolute(s3)
                theta_index = np.argmin(s3)  # finds index of theta that's closest to theta from motion model 
                    
                
                bel[x_i][y_i][theta_index] += bel[i][j][k]
                
                bel_bar[i][j][k] =  temp_bel + bel[x_i][y_i][theta_index]*bel[i][j][k]
                temp_bel += bel[x_i][y_i][theta_index]*bel[i][j][k]

                measure = observe2(measurement,x_i,y_i,theta_mot)

                bel[i][j][k] = measure * bel_bar[i][j][k]
    bel_sum = 0
                
    for q in range(bel.shape[0]):
        for w in range(bel.shape[1]):
            for e in range(bel.shape[2]):
                bel_sum += bel[q][w][e]  # normalizing belief
    bel = bel/bel_sum
        
                        
    return bel



# for problem 1
for i in range(6):
    bel = bayes_filter1(range1[i],bel)

    #to find the maximum value in belief matrix
    h = 0
    for p in range(bel.shape[0]):
            for j in range(bel.shape[1]):
                for k in range(bel.shape[2]):
                    if bel[p][j][k] > h:
                        x = p
                        y = j
                        h = bel[p][j][k]
                        theta = theta_actual[k]
    theta_degree = (180/np.pi)*theta
    print(x,' ',y,' ',theta_degree)
    final_x1[i] = x
    final_y1[i] = y
    final_theta1[i] = theta_degree
    
    
    
# for problem 2    
for i in range(6):
    bel2 = bayes_filter2(range2[i],bel2)

    #to find the maximum value in belief matrix
    h = 0
    for p in range(bel2.shape[0]):
            for j in range(bel2.shape[1]):
                for k in range(bel2.shape[2]):
                    if bel2[p][j][k] > h:
                        x = p
                        y = j
                        h = bel2[p][j][k]
                        theta = theta_actual[k]
    theta_degree = (180/np.pi)*theta
    print(x,' ',y,' ',theta_degree)
    final_x2[i] = x
    final_y2[i] = y
    final_theta2[i] = theta_degree
    
    
    
    
mp.title('Plot for Distance to Wall - Question 1')
mp.xlabel('time steps')
mp.ylabel('distance')
mp.plot(x_k,final_y1)
mp.show()

mp.title('Plot for Angle to Wall - Question 1')
mp.xlabel('time steps')
mp.ylabel('Angle in degrees')
mp.plot(x_k,final_theta1)
mp.show()

mp.title('Plot for Distance to Wall - Question 2')
mp.xlabel('time steps')
mp.ylabel('distance')
x_k = [0.2,0.4,0.6,0.8,1.0,1.2]
mp.plot(x_k,final_y2)
mp.show()

mp.title('Plot for Angle to Wall - Question 2')
mp.xlabel('time steps')
mp.ylabel('Angle in degrees')
mp.plot(x_k,final_theta2)
mp.show()

print(final_x1,' ',final_y1,' ', final_theta1)
print(final_x2,' ',final_y2,' ',final_theta2)

f = open( 'result.py', 'w' )
f.write( 'final_x1 = ' + repr(final_x1) + '\n' )
f.write( 'final_y1 = ' + repr(final_y1) + '\n' )
f.write( 'final_theta1 = ' + repr(final_theta1) + '\n' )
f.write( 'final_x2 = ' + repr(final_x2) + '\n' )
f.write( 'final_y2 = ' + repr(final_y2) + '\n' )
f.write( 'final_theta2 = ' + repr(final_theta2) + '\n' )
f.close()
    