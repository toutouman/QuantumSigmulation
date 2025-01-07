# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:09:35 2021

@author: 馒头你个史剃磅
"""


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
starttime=int(time.time())

N=30
C_q=40e-15
e=1.60217e-19
h=6.62607e-34
k_b= 1.3806505e-23
T_eff=20e-3
h_bar=h/2/np.pi
# E_C=e**2/(2*C_q)/h_bar/1e9
E_C=0.21*2*np.pi  #GHZ
E_J=12*2*np.pi #GHZ
num=201
sigma_0=20*0.15
g=1/np.sqrt(2*np.pi)/sigma_0*1.019 #0.0125*2*np.pi,1.019来自Omega的偏差

T=40 #18结果更好
t_range=np.linspace(0,T/2,num) #ns
N_g=np.linspace(-2,2,num)
n_range=np.linspace(0,N,N+1)
n_g=0.5

Omega=(E_J/E_C/8)**0.25/np.sqrt(2)
lambda_0=0.5
t_0=T/4


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

def Omega_I(t,arg):
    phi_0=arg['phi_0']
    s_t=g/Omega*np.exp(-(t-t_0)**2/(2*sigma_0**2))*np.pi/2
    omega_t=s_t*np.sin(omega_0*(t+phi_0))
    return omega_t
def Omega_Q(t,arg):
    phi_0=arg['phi_0']
    s_dt=-g/Omega*np.exp(-(t-t_0)**2/(2*sigma_0**2))*(t-t_0)/sigma_0**2*np.pi/2
    omega_t=-(lambda_0*s_dt/alpha)*np.cos(omega_0*(t+phi_0))
    return omega_t


E_0=H(H_j,n_g).eigenenergies()[0]
E_1=H(H_j,n_g).eigenenergies()[1]
E_2=H(H_j,n_g).eigenenergies()[2]
psi_0=H(H_j,n_g).eigenstates()[1][0]
psi_1=H(H_j,n_g).eigenstates()[1][1]
psi_2=H(H_j,n_g).eigenstates()[1][2]
rho_0=psi_0*psi_0.dag()
rho_1=psi_1*psi_1.dag()
rho_2=psi_2*psi_2.dag()
rho_z=psi_0*psi_0.dag()-psi_1*psi_1.dag()
rho_x=psi_0*psi_1.dag()+psi_1*psi_0.dag()
rho_y=-1j*psi_0*psi_1.dag()+1j*psi_1*psi_0.dag()
omega_0=E_1-E_0
omega_1=E_2-E_1
omega_02=(omega_0+omega_1)/2
alpha=omega_0-omega_1
print(omega_0/2/np.pi)
#%%
H_t=[H(H_j,n_g),[H_n(H_j),Omega_I],[H_n(H_j),Omega_Q]]

arg_1={'phi_0':0}
result_0=mesolve(H_t,psi_0,t_range,[] ,[],args=arg_1)
psi_t0=result_0.states[-1]

arg_2={'phi_0':T/2}
result=mesolve(H_t,psi_t0,t_range,[],[],
                args=arg_2)
P_0=[]
P_1=[]
P_2=[]
P_x=[]
P_y=[]
P_z=[]
for i in range(num):
    psi_i=result_0.states[i]
    rho_i=result_0.states[i]*result_0.states[i].dag()
    t_i=t_range[i]
    U_i=np.exp(-1j*omega_0*(t_i))
    U_j=np.exp(1j*omega_0*(t_i))
    sigma_xt=U_i*psi_1*psi_0.dag()+U_j*psi_0*psi_1.dag()
    sigma_yt=1j*U_i*psi_1*psi_0.dag()-1j*U_j*psi_0*psi_1.dag()
    sigma_zt=rho_z
    P_x.append((rho_i*sigma_xt).tr())
    P_y.append((rho_i*sigma_yt).tr())
    P_z.append((rho_i*sigma_zt).tr())
    P_0.append((rho_i*rho_0).tr())
    P_1.append((rho_i*rho_1).tr())
    P_2.append((rho_i*rho_2).tr())
for i in range(num):
    psi_i=result.states[i]
    rho_i=result.states[i]*result.states[i].dag()
    t_i=t_range[i]
    U_i=np.exp(-1j*omega_0*(t_i+T/2))
    U_j=np.exp(1j*omega_0*(t_i+T/2))
    sigma_xt=U_i*psi_1*psi_0.dag()+U_j*psi_0*psi_1.dag()
    sigma_yt=1j*U_i*psi_1*psi_0.dag()-1j*U_j*psi_0*psi_1.dag()
    sigma_zt=rho_z
    P_x.append((rho_i*sigma_xt).tr())
    P_y.append((rho_i*sigma_yt).tr())
    P_z.append((rho_i*sigma_zt).tr())
    P_0.append((rho_i*rho_0).tr())
    P_1.append((rho_i*rho_1).tr())
    P_2.append((rho_i*rho_2).tr())
    
# P_0=result.expect[0]
# P_1=result.expect[1]
# P_2=result.expect[2]

# plt.figure()
# plt.plot(t_range,P_0,label=r'$P_0$')
# plt.plot(t_range,P_1,label=r'$P_1$')
# plt.plot(t_range,P_2,label=r'$P_2$')
# plt.legend()

#waveform envelope of signalt
plt.figure()
plt.plot(t_range,[np.exp(-(t_range[i]-t_0)**2/(2*sigma_0**2))*np.pi
                  for i in range(num)],'b',label='I')
plt.plot(t_range,[-lambda_0/alpha*np.exp(-(t_range[i]-t_0)**2/(2*sigma_0**2))
                  *(t_range[i]-t_0)/sigma_0**2*np.pi
                  for i in range(num)],'r--',label='Q')
plt.show()
plt.legend()

P_b=[P_x,P_y,P_z]  

b=Bloch()
b.add_points(P_b,'l')
b.show()  


endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')
