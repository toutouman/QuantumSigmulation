# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 23:21:01 2021

@author: 馒头你个史剃磅
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
starttime=int(time.time())

N=15
E_C1=0.225*2*np.pi  #GHZ
E_C2=0.183*2*np.pi
E_J1=19*2*np.pi #GHZ
E_J2=19*2*np.pi #GHZ
num=301
phi_e=0
g=0.0125*2*np.pi

T=40
t_range=np.linspace(0,T,num) #ns
n_g=0
n_range=np.linspace(0,N,N+1)
Phi_e=np.linspace(0,0.3,num)

def H_1(E_C):
    H_j1=tensor(-E_J1/2*(qdiags([1]*(N-1),1)+qdiags([1]*(N-1),-1)),qeye(N))
    H_t=H_j1
    for i in range(N):
        n=int(n_range[i])
        H_c=tensor((4*E_C*(n-n_g-(N/2))**2)*basis(N,n)*basis(N,n).dag(),qeye(N))
        H_t=H_t+H_c
    return(H_t)

def H_2(E_C):
    H_j2=tensor(qeye(N),-E_J2/2*(qdiags([1]*(N-1),1)+qdiags([1]*(N-1),-1)))
    H_t=H_j2
    for i in range(N):
        n=int(n_range[i])
        H_c=tensor(qeye(N),(4*E_C*(n-n_g-(N/2))**2)*basis(N,n)*basis(N,n).dag())
        H_t=H_t+H_c
    return(H_t)

def H_n():
    H_t=0
    for i in range(N):
        n=int(n_range[i])
        H_c=(n-(N/2))*basis(N,n)*basis(N,n).dag()
        H_t=H_t+H_c
    return(H_t)
psi_00=(H_1(E_C1)+H_2(E_C2)).eigenstates()[1][0]
psi_01=(H_1(E_C1)+H_2(E_C2)).eigenstates()[1][1]
psi_10=(H_1(E_C1)+H_2(E_C2)).eigenstates()[1][2]
psi_11=(H_1(E_C1)+H_2(E_C2)).eigenstates()[1][3]
rho_00=psi_00*psi_00.dag()
rho_01=psi_01*psi_01.dag()
rho_10=psi_10*psi_10.dag()
rho_11=psi_11*psi_11.dag()

H=H_1(E_C1)+H_2(E_C2)+g*tensor(H_n(),H_n())
result=mesolve(H,psi_01,t_range,[],[])

omega_1=H_1(E_C1).eigenenergies()[16]-H_1(E_C1).eigenenergies()[0]
omega_2=H_2(E_C2).eigenenergies()[16]-H_2(E_C2).eigenenergies()[0]
P_00=[]
P_01=[]
P_10=[]
P_11=[]
for i in range(num):
    t_i=t_range[i]
    psi_t=result.states[i]
    rho_t=psi_t*psi_t.dag()
    P_00.append((rho_00*rho_t).tr())
    P_01.append((rho_01*rho_t).tr())
    P_10.append((rho_10*rho_t).tr())
    P_11.append((rho_11*rho_t).tr())

plt.figure()
plt.plot(t_range, P_00,label='P_00')
plt.plot(t_range, P_01,label='P_01')
plt.plot(t_range, P_10,label='P_10')
plt.plot(t_range, P_11,label='P_11')
plt.legend()
plt.show()




