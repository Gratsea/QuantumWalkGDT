#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forward quantum walk with specific coin operators

@author: katerina
"""


import math
import numpy as np
import cmath

#define final state
n=3 #number os steps
k=n+1 #number of sites

initial = np.zeros((2*k,1),dtype=complex)
initial[0][0]= 1
initial[1][0]= 1.5
initial /= np.linalg.norm(initial)


Initial = initial

#definition of invS
invS = np.zeros((2*k,2*k),dtype=complex)
matrixS = np.zeros((2*k,2*k),dtype=complex)
for i in range (0,2*k,2) :
    invS[0+i][0+i] =1.
    matrixS[0+i][0+i] =  1.
    if (i+3)< 2*k :
        invS[1+i][3+i] = 1. #S-1
        matrixS[3+i][1+i] = 1.
  
listSt=[]      
'''listc = [  5.02960802e-02,  -8.28193118e-01,   3.24132166e+00,
        -2.43973068e-01,   9.16008133e-01,   2.32160702e+00,
        -7.25310961e-09,   6.73524945e+00,   1.05685586e+00 ]  '''

listc=[  3.03165347e-01 , -5.57101025e-01 ,  2.62601578e+00,   2.43974271e-01 ,  1.73657437e+00,   1.61072078e+00,  -5.44413974e-10,   6.31063138e+00,  -7.86446152e-02]      

l=0
for j in range (0,n,+1) : 
    
    z=listc
    
    x=abs(z[0+l])
    y=z[1+l]
    v=z[2+l]

    c=np.zeros((2,2),dtype=complex)

    c[0][0]=   math.sqrt(x)
    c[0][1]= (math.sqrt(1-x)) * (math.cos(y*math.pi) + math.sin(y*math.pi)*1j) 
    c[1][0]= (math.sqrt(1-x)) * (math.cos(v*math.pi) + math.sin(v*math.pi)*1j)         
    c[1][1]= -(math.sqrt(x)) * (math.cos((y+v)*math.pi) + math.sin((y+v)*math.pi)*1j)  
   
    matrixC = np.zeros((2*k,2*k),dtype=complex)
    for i in range (0,2*k,2):
        matrixC[0+i][0+i] = c[0][0]
        matrixC[1+i][1+i] = c[1][1]
        matrixC[0+i][1+i] = c[0][1]          
        matrixC[1+i][0+i] = c[1][0] 
        
    m1 = np.dot(matrixC,initial)
    m2 = np.dot(matrixS,m1)
    print (m2)
    '''m2 /= np.linalg.norm(m2)
    print (m2)'''

    listSt.append (m2)
    initial = m2
    l+=3
    if (j==n-1) :
        Phi=m2
        print (Phi)


Phi_target= np.array([[ 0.00],       [ 0.       ],       [ 0.6148   ],       [ 0.3492   ],       [ 0.3493   ],       [-0.6148   ],       [ 0.       ],       [ 0.00]])
Fidelity = cmath.polar(np.dot(Phi.transpose(),Phi_target))
Fidelity = Fidelity[0]