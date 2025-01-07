# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:24:08 2021

@author: 馒头你个史剃磅
"""


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
starttime=int(time.time())
N=20
E_C=1
E_J=100*E_C
gamma=0.8
phi_a=0
num=101
n_g=0
Phi_a=np.linspace(0,1,num)
n_range=np.linspace(0,N,N+1)

def H_1(gamma):
    H_j1=tensor(-E_J/2*(qdiags([1]*(N-1),1)+qdiags([1]*(N-1),-1)),qeye(N))
    H_t=H_j1
    for i in range(N):
        n=int(n_range[i])
        H_c=tensor((4*E_C*(n-n_g-(N/2))**2)*basis(N,n)*basis(N,n).dag(),qeye(N))
        H_t=H_t+H_c
    return(H_t)

def H_2(gamma):
    H_j2=tensor(qeye(N),-E_J/2*(qdiags([1]*(N-1),1)+qdiags([1]*(N-1),-1)))
    H_t=H_j2
    for i in range(N):
        n=int(n_range[i])
        H_c=tensor(qeye(N),(4*E_C*(n-n_g-(N/2))**2)*basis(N,n)*basis(N,n).dag())
        H_t=H_t+H_c
    return(H_t)
def H_3(gamma,phi_a):
    H_j3=-gamma*E_J/2*(
        np.exp(2j*np.pi*phi_a)*tensor(qdiags([1]*(N-1),1),qdiags([1]*(N-1),-1))+
        np.exp(-2j*np.pi*phi_a)*tensor(qdiags([1]*(N-1),-1),qdiags([1]*(N-1),1)))
    H_t=H_j3
    for i in range(N):
        n_i=int(n_range[i])
        for j in range(N):
            n_j=int(n_range[j])
            psi_ij=tensor(basis(N,n_i),basis(N,n_j))
            H_c=4*E_C/gamma*((n_i+n_j-n_g-N)**2)*psi_ij*psi_ij.dag()
            H_t=H_t+H_c
    return(H_t)

def multy_job(k):
    phi_a=Phi_a[k]
    H_t=H_1(gamma)+H_2(gamma)+H_3(gamma,phi_a)
    H_t0=H_1(gamma)+H_2(gamma)+H_3(gamma,0.5)
    E_01=(H_t0.eigenenergies()[1]-H_t0.eigenenergies()[0])
    E_0=H_t0.eigenenergies()[0]
    e_i=H_t.eigenenergies()
    e_0=((e_i[0])-E_0)/E_01
    e_1=((e_i[1])-E_0)/E_01
    e_2=((e_i[2])-E_0)/E_01
    e_3=((e_i[3])-E_0)/E_01
    e_4=((e_i[4])-E_0)/E_01
    return(e_0,e_1,e_2,e_3,e_4)
data = Parallel(n_jobs=3, verbose=2)(delayed(multy_job)(k) for k in range(num))

E_0=[i[0] for i in data]
E_1=[i[1] for i in data]
E_2=[i[2] for i in data]
E_3=[i[3] for i in data]
E_4=[i[4] for i in data]
Omega_01=[i[1]-i[0] for i in data]
Omega_12=[i[2]-i[1] for i in data]
Omega_13=[i[3]-i[1] for i in data]



plt.plot(Phi_a,E_0,label=r'$E_0$')
plt.plot(Phi_a,E_1,label=r'$E_1$')
plt.plot(Phi_a,E_2,label=r'$E_2$')
plt.plot(Phi_a,E_3,label=r'$E_3$')
plt.plot(Phi_a,E_4,label=r'$E_4$')
# plt.plot(N_g,E_0sim,label=r'$E_0$ by Perturbation')
# plt.plot(N_g,E_1sim,label=r'$E_1$ by Perturbation')
# plt.plot(N_g,E_2sim,label=r'$E_2$ by Perturbation')
plt.ylabel(r'Energy level')
plt.xlabel(r'Bias flux $\Phi_e/\Phi_0$')
plt.legend()

plt.figure()
plt.plot(Phi_a,Omega_01,label=r'$\omega_{01}$')
plt.plot(Phi_a,Omega_12,label=r'$\omega_{12}$')
plt.plot(Phi_a,Omega_13,label=r'$\omega_{11}$')
plt.legend()



endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')