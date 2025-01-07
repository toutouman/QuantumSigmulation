# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:27:59 2021

@author: 馒头你个史剃磅
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
starttime=int(time.time())

N=30
E_C=0.225*2*np.pi  #GHZ
E_J1=22*2*np.pi #GHZ
E_J2=19*2*np.pi #GHZ
num=201
phi_e=0
g=0.0125*2*np.pi

T=40
t_range=np.linspace(0,T,num) #ns
n_g=0
n_range=np.linspace(0,N,N+1)
Phi_e=np.linspace(0,0.3,num)

def H_1(E_J):
    H_j1=tensor(-E_J/2*(qdiags([1]*(N-1),1)+qdiags([1]*(N-1),-1)),qeye(N))
    H_t=H_j1
    for i in range(N):
        n=int(n_range[i])
        H_c=tensor((4*E_C*(n-n_g-(N/2))**2)*basis(N,n)*basis(N,n).dag(),qeye(N))
        H_t=H_t+H_c
    return(H_t)

def H_2(E_J):
    H_j2=tensor(qeye(N),-E_J/2*(qdiags([1]*(N-1),1)+qdiags([1]*(N-1),-1)))
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

H=H_1(E_J1)+H_2(E_J2)
psi_00=H.eigenstates()[1][0]
psi_01=H.eigenstates()[1][1]
psi_10=H.eigenstates()[1][2]

def multy_job(k):
    phi_e=Phi_e[k]
    E_J1t=np.cos(phi_e*np.pi)*E_J1
    H_k=H_1(E_J1t)+H_2(E_J2)+g*tensor(H_n(),H_n())
    # E_01=(H(gamma,0).eigenenergies()[1]-H(gamma,0).eigenenergies()[0])
    # E_0=H(gamma,phi_a).eigenenergies()[0]
    e_i=H_k.eigenenergies()
    e_0=(e_i[0])/2/np.pi
    e_1=(e_i[1])/2/np.pi
    e_2=(e_i[2])/2/np.pi
    e_3=(e_i[3])/2/np.pi
    e_4=(e_i[4])/2/np.pi
    e_5=(e_i[5])/2/np.pi
    return(e_0,e_1,e_2,e_3,e_4,e_5)
data = Parallel(n_jobs=3, verbose=2)(delayed(multy_job)(k) for k in range(num))

E_0=[i[0] for i in data]
E_1=[i[1] for i in data]
E_2=[i[2] for i in data]
E_3=[i[3] for i in data]
E_4=[i[4] for i in data]
E_5=[i[5] for i in data]
Omega_01=[i[1]-i[0] for i in data]
Omega_02=[i[2]-i[0] for i in data]
Omega_03=[i[3]-i[0] for i in data]
Omega_04=[i[4]-i[0] for i in data]
Omega_05=[i[5]-i[0] for i in data]
Delta_12=[Omega_02[i]-Omega_01[i] for i in range(num)]
Delta_34=[Omega_04[i]-Omega_03[i] for i in range(num)]
Delta_45=[Omega_05[i]-Omega_04[i] for i in range(num)]

plt.figure()
plt.plot(Phi_e,E_0,label=r'$E_0$')
plt.plot(Phi_e,E_1,label=r'$E_1$')
plt.plot(Phi_e,E_2,label=r'$E_2$')
plt.plot(Phi_e,E_3,label=r'$E_3$')
plt.legend()

plt.figure()
plt.plot(Phi_e,Omega_01,label=r'$\omega_{1}$')
plt.plot(Phi_e,Omega_02,label=r'$\omega_{2}$')
plt.plot(Phi_e,Omega_03,label=r'$\omega_{3}$')
plt.plot(Phi_e,Omega_04,label=r'$\omega_{4}$')
plt.plot(Phi_e,Omega_05,label=r'$\omega_{5}$')
plt.legend()  

plt.figure()
plt.plot(Phi_e,Delta_45)
plt.plot(Phi_e,Delta_34)
plt.legend()  

endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')