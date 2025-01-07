# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:53:24 2021

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
E_J=100*E_C #19*2*np.pi #GHZ
num=301
sigma_0=3
g=1/np.sqrt(2*np.pi)/sigma_0*1.020 #0.0125*2*np.pi,1.020来自Omega的偏差

T=30
t_range=np.linspace(0,T,num) #ns
# n_g=0.5
N_g=np.linspace(-2,2,num)
n_range=np.linspace(0,N,N+1)
H_j=-E_J/2*(qdiags([1]*(N-1),1)+qdiags([1]*(N-1),-1))
def H(H_j,n_g):
    H_t=H_j
    for i in range(N):
        n=int(n_range[i])
        H_c=(4*E_C*(n-n_g-(N/2))**2)*basis(N,n)*basis(N,n).dag()
        H_t=H_t+H_c
    return(H_t)
def H_n(H_j):
    H_t=0
    for i in range(N):
        n=int(n_range[i])
        H_c=(n-(N/2))*basis(N,n)*basis(N,n).dag()
        H_t=H_t+H_c
    return(H_t)


E_0=H(H_j,0).eigenenergies()[0]
E_1=H(H_j,0).eigenenergies()[1]
E_2=H(H_j,0).eigenenergies()[2]
psi_0=H(H_j,0).eigenstates()[1][0]
psi_1=1j*H(H_j,0).eigenstates()[1][1]
psi_2=H(H_j,0).eigenstates()[1][2]
omega_0=E_1-E_0
omega_1=E_2-E_1
alpha=omega_0-omega_1

# def Omega(t,arg):
#     omega_t=0.0125*4*np.pi*np.exp(-t**2/(2*12.7886**2))*np.sin(omega_0*t)
#     return omega_t 

Omega=(E_J/E_C/8)**0.25/np.sqrt(2)
lambda_0=0.2
t_0=T/2
def Omega_I(t,arg):
    s_t=g/Omega*np.exp(-(t-t_0)**2/(2*sigma_0**2))*np.pi
    omega_t=s_t*np.sin(omega_0*t)
    return omega_t
def Omega_Q(t,arg):
    s_dt=-g/Omega*np.exp(-(t-t_0)**2/(2*sigma_0**2))*(t-t_0)/sigma_0**2*np.pi
    omega_t=-(lambda_0*s_dt/alpha)*np.cos(omega_0*t)
    return omega_t

rho_0=psi_0*psi_0.dag()
rho_1=psi_1*psi_1.dag()
rho_2=psi_2*psi_2.dag()
sigma_z=rho_0-rho_1
sigma_x=psi_1*psi_0.dag()+psi_0*psi_1.dag()
sigma_y=1j*psi_1*psi_0.dag()-1j*psi_0*psi_1.dag()
H=[H(H_j,0),[H_n(H_j),Omega_I],[H_n(H_j),Omega_Q]]
result=mesolve(H,psi_0,t_range,[],[])
P_x=[]
P_y=[]
P_z=[]
for i in range(num):
    rho_i=result.states[i]*result.states[i].dag()
    t_i=t_range[i]
    U_i=np.exp(-1j*omega_0*t_i)
    U_j=np.exp(1j*omega_0*t_i)
    sigma_xt=U_i*psi_1*psi_0.dag()+U_j*psi_0*psi_1.dag()
    sigma_yt=1j*U_i*psi_1*psi_0.dag()-1j*U_j*psi_0*psi_1.dag()
    sigma_zt=sigma_z
    P_x.append((rho_i*sigma_xt).tr())
    P_y.append((rho_i*sigma_yt).tr())
    P_z.append((rho_i*sigma_zt).tr())
    # P_b.append([P_x[i],P_y[i],P_z[i]])
    

# P_0=result.expect[0]
# P_1=result.expect[1]
# P_2=result.expect[2]


plt.figure()
plt.plot(t_range,P_x,label=r'$P_y$')
plt.plot(t_range,P_y,label=r'$P_x$')
plt.plot(t_range,P_z,label=r'$P_z$')
plt.legend()

P_b=[P_x,P_y,P_z]  

b=Bloch()
b.add_points(P_b,'l')
b.show()  

plt.figure()
plt.plot(t_range,[np.exp(-(t_range[i]-t_0)**2/(2*sigma_0**2))*np.pi for i in range(num)])


endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')
