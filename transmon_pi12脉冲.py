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
E_C=0.0932*2*np.pi  #GHZ
E_J=50*2*np.pi #GHZ
num=201
sigma_0=3
g=1/np.sqrt(2*np.pi)/sigma_0*1.019 #0.0125*2*np.pi,1.019来自Omega的偏差

T=24 #18结果更好
t_range=np.linspace(0,T,num) #ns
N_g=np.linspace(-2,2,num)
n_range=np.linspace(0,N,N+1)
n_g=0.5

Omega=(E_J/E_C/8)**0.25/np.sqrt(2)
lambda_0=0.5/2.976
t_0=T/2


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
    s_t=g/Omega*np.exp(-(t-t_0)**2/(2*sigma_0**2))*np.pi/1.36895/2
    omega_t=s_t*np.sin(omega_1*(t+phi_0))
    return omega_t
def Omega_Q(t,arg):
    phi_0=arg['phi_0']
    s_dt=-g/Omega*np.exp(-(t-t_0)**2/(2*sigma_0**2))*(t-t_0)/sigma_0**2*np.pi/1.36895/2
    omega_t=-(lambda_0*s_dt/alpha)*np.cos(omega_1*(t+phi_0))
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
omega_0=E_1-E_0
omega_1=E_2-E_1
omega_02=(omega_0+omega_1)
alpha=omega_0-omega_1
print(omega_0/2/np.pi)
#%%
H=[H(H_j,n_g),[H_n(H_j),Omega_I],[H_n(H_j),Omega_Q]]

arg_1={'phi_0':0}
result_0=mesolve(H,psi_1,t_range,[] ,[],args=arg_1)
psi_t0=result_0.states[-1]

arg_2={'phi_0':T}
result=mesolve(H,psi_t0,t_range,[],[rho_0,rho_1,rho_2],
                args=arg_2)


P_0=result.expect[0]
P_1=result.expect[1]
P_2=result.expect[2]

plt.figure()
plt.plot(t_range,P_0,label=r'$P_0$')
plt.plot(t_range,P_1,label=r'$P_1$')
plt.plot(t_range,P_2,label=r'$P_2$')
plt.legend()

#waveform envelope of signal
plt.figure()
plt.plot(t_range,[np.exp(-(t_range[i]-t_0)**2/(2*sigma_0**2))*np.pi
                  for i in range(num)],'b',label='I')
plt.plot(t_range,[-lambda_0/alpha*np.exp(-(t_range[i]-t_0)**2/(2*sigma_0**2))
                  *(t_range[i]-t_0)/sigma_0**2*np.pi
                  for i in range(num)],'r--',label='Q')
plt.show()
plt.legend()
print(P_2[-1])
#%%
Gamma_1=1/100
Gamma_2=2*Gamma_1
Gamma_phi=1/20
d_omega=0.0

def T2_delay_22(t,d_omega):
    rho_ii=(0.25+Gamma_2*0.25/(Gamma_2-Gamma_1))*np.exp(-Gamma_1*t)-Gamma_1*0.25/(Gamma_2-Gamma_1)*np.exp(-Gamma_2*t)
    delay=rho_ii+0.5*np.exp(-(Gamma_1+Gamma_2)/2*t)*np.cos(2*np.pi*d_omega*t)
    return(delay)
def T2_delay_11(t,d_omega):
    rho_ii=(0.25+Gamma_2*0.25/(Gamma_2-Gamma_1))*np.exp(-Gamma_1*t)-Gamma_1*0.25/(Gamma_2-Gamma_1)*np.exp(-Gamma_2*t)
    delay=rho_ii-0.5*np.exp(-(Gamma_1+Gamma_2)/2*t)*np.cos(2*np.pi*d_omega*t)
    return(delay)

time_r=np.linspace(0,100,1001)

plt.figure()
plt.plot(time_r,[T2_delay_22(i,d_omega) for i in time_r],label='P_22')
plt.plot(time_r,[T2_delay_11(i,d_omega) for i in time_r],label='P_11')
plt.plot(time_r,[1-T2_delay_11(i,d_omega)-T2_delay_22(i,d_omega) for i in time_r],label='P_00')
plt.legend()
    

endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')