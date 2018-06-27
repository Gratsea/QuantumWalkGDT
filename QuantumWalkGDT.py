# -*- coding: utf-8 -*-
"""
Quantum walk with GDT. 3 free parameters for each coin operator

@author: kgratsea
"""

import numpy as np
import cmath
import math
from scipy import optimize
global final
import random 

def tensor(vectorA,vectorB) :
    m = np.size(vectorA,0)
    n = np.size(vectorB,0)
    tens=np.zeros((m,n))
    for i in range(m) :
        for j in range(n) :
            tens[i][j] = vectorA[i]*vectorB[j]
    return (tens);

class MyTakeStep(object):
       def __init__(self, stepsize=0.1):
           self.stepsize = stepsize
       def __call__(self, x):
           s = self.stepsize
           x[0] += np.random.uniform(-0.01, 0.01)
           x[1:] += np.random.uniform(-0.02,0.02, x[1:].shape)
           return x


def func(z) :    
    n=3 #number of steps
    k=n+1 #number of sites at the final state
    
    initial = np.zeros((2*k,1),dtype=complex)
    
    initial[0][0]= 1
    initial[1][0]= 1.5
    initial/= np.linalg.norm(initial)
    
    Initial = initial
    #print (Initial)
    
    f = open("test_GDT.txt","a+")
    f.write("Initial")
    f.close()
    with open('test_GDT.txt', 'a+') as f:
       print (Initial,file=f)
    f.close()
   
    
    #definition of invS
    invS = np.zeros((2*k,2*k),dtype=complex)
    matrixS = np.zeros((2*k,2*k),dtype=complex)
    for i in range (0,2*k,2) :
        invS[0+i][0+i] =1.
        matrixS[0+i][0+i] =  1.
        if (i+3)< 2*k :
            invS[1+i][3+i] = 1. #S-1
            matrixS[3+i][1+i] = 1.
    
    listSt = []
    listc = []
    listC = []

    listSt.append (initial)
    
    #Define coin operators with gdt
    
    l = 0 # for corresponding the correct coin parameters at each step n
    for j in range (0,n,+1) : 
        print ("n",j)
        c=np.zeros((2,2),dtype=complex)
        x=z[0+l]
        y=z[1+l]
        v=z[2+l]
        if (1-x<0): 
            print ("here1")
            x /= 10.
            print (x)
        if (y>2*math.pi): 
            print ("here2")
            y /= 10.
            print (y)
        if (v>2*math.pi): 
            print ("here3")
            v /= 10.
            print (v)
        
        c[0][0]=   math.sqrt(x)
        c[0][1]= (math.sqrt(1-x)) * (math.cos(y) + math.sin(y)*1j) 
        c[1][0]= (math.sqrt(1-x)) * (math.cos(v) + math.sin(v)*1j)         
        c[1][1]= -(math.sqrt(x)) * (math.cos(y+v) + math.sin(y+v)*1j)  
        listc.append(c)
        matrixC = np.zeros((2*k,2*k),dtype=complex)
        print (c)
        
        for i in range (0,2*k,2):
            matrixC[0+i][0+i] = c[0][0]
            matrixC[1+i][1+i] = c[1][1]
            matrixC[0+i][1+i] = c[0][1]          
            matrixC[1+i][0+i] = c[1][0]   
         
        listC.append (matrixC)    
        
        
        m1 = np.dot(matrixC,initial)
        m2 = np.dot(matrixS,m1)   #next state
        print (m2)
        listSt.append (m2)
        initial = m2/np.linalg.norm(m2)
        l += 3 # moving to the next coin parameters
        
    Phi=initial    
    print ("Phi",Phi)
    Phi_target= np.array([[ 0.0066874],       [ 0.       ],       [ 0.6148   ],       [ 0.3492   ],       [ 0.3493   ],       [-0.6148   ],       [ 0.       ],       [ 0.0067874]])
    print ("Phi_target",Phi_target)

    kate = cmath.polar(np.dot(Phi.transpose(),Phi_target))
    Fidelity=math.pow(kate[0],2)
    #Fidelity = np.dot(Phi.transpose(),Phi_target)*np.dot(Phi.transpose(),Phi_target)
    print ("Fidelity",Fidelity)

    f = open("test_GDT.txt","a+")
    f.write("1-Fidelity")
    f.close()
    with open('test_GDT.txt', 'a+') as f:
        print (1-Fidelity.real,file=f)
    f.close()
    print ("1-Fidelity",1-Fidelity,z)    
    return (1-Fidelity)
        
   
initial_coin_parameters=[1/2,0,math.pi,1/4,math.pi/2.,math.pi/2.,1/8,2*math.pi,0]    
'''
f=3
for i in range (1,3*f,3):    
    initial_coin_parameters.append(random.uniform(0,1))   
    initial_coin_parameters.append(random.uniform(0,2*math.pi))   
    initial_coin_parameters.append(random.uniform(0,2*math.pi))   
    '''
Initial_coin_par=initial_coin_parameters  
        
minimizer_kwargs = {"method": "BFGS"}
mytakestep = MyTakeStep()
#ret = optimize.basinhopping(func,x0,niter=1 ,T=1.0, stepsize=0.5, minimizer_kwargs=minimizer_kwargs,interval = 5, niter_success = 6 )
#ret = optimize.basinhopping(func,initial_coin_parameters,take_step=mytakestep,  minimizer_kwargs=minimizer_kwargs,niter=20, T=1.0, disp = True )
ret = optimize.basinhopping(func,initial_coin_parameters, minimizer_kwargs=minimizer_kwargs,niter=3, T=1.0, disp = True )
  