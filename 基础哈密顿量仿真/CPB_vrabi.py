# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:01:12 2020

@author: 馒头你个史剃磅
"""


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
starttime=int(time.time())
N=50
E_C=1
E_J=5*E_C
num=81
# n_g=0.5
n_g=0.5
n_range=np.linspace(0,N,N+1)
H_j=-E_J/2*(qdiags([1]*(N-1),1)+qdiags([1]*(N-1),-1))
def H(H_j,n_g):
    H_t=H_j
    for i in range(N):
        n=int(n_range[i])
        H_c=(4*E_C*(n-n_g-(N/2))**2)*basis(N,n)*basis(N,n).dag()
        H_t=H_t+H_c
    return(H_t)

H_t=H(H_j,n_g)
e_i=H_t.eigenenergies()
e_0=e_i[0]
e_1=e_i[1]
e_2=e_i[2]
e_3=e_i[3]
omega_01=e_1-e_0

time=np.linspace(0,50,1000)
psi_0=H(H_j,0.5).eigenstates()[1][0]
psi_1=H(H_j,0.5).eigenstates()[1][1]
psi_2=H(H_j,0.5).eigenstates()[1][2]
psi_3=H(H_j,0.5).eigenstates()[1][3]
psi_p=(psi_0+psi_1).unit()

H_N=0
for i in range(N):
    n=int(n_range[i])
    H_N=H_N+n*basis(N,n)*basis(N,n).dag()*0.15
def RF_x(t,arg):
    return(np.cos((omega_01)*t))
H_x=[H(H_j,0.5),[H_N,RF_x]]

result=mesolve(H_x,psi_0,time,[],[])
F_0=[]
F_1=[]
F_2=[]
F_3=[]
for i in range(1000):
    psi_t=result.states[i]
    F_0.append(abs((psi_0.dag()*psi_t)[0][0]))
    F_1.append(abs((psi_1.dag()*psi_t)[0][0]))
    F_2.append(abs((psi_2.dag()*psi_t)[0][0]))
    F_3.append(abs((psi_3.dag()*psi_t)[0][0]))
plt.figure()
plt.plot(time,F_0,label='F_0')
plt.plot(time,F_1,label='F_1')
plt.plot(time,F_2,label='F_2')
plt.plot(time,F_3,label='F_3')
plt.legend()