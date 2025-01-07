# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 22:27:52 2021

@author: 馒头你个史剃磅
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
starttime=int(time.time())

omega_0=5*2*np.pi
sigma_0=2
g=1/np.sqrt(2*np.pi)/sigma_0/2
T=40
num=101
t_range=np.linspace(0,T,num)

psi_0=basis(2,0)
psi_1=basis(2,1)
rho_0=psi_0*psi_0.dag()
rho_1=psi_1*psi_1.dag()

H_0=-0.5*omega_0*sigmaz()
lambda_0=0.5
t_0=T/2
def Omega_I(t,arg):
    s_t=g*np.exp(-(t-t_0)**2/(2*sigma_0**2))*np.pi
    omega_t=s_t*np.sin(omega_0*t)
    return omega_t
# def Omega_Q(t,arg):
#     s_dt=-g*np.exp(-(t-t_0)**2/(2*sigma_0**2))*(t-t_0)/sigma_0**2*np.pi
#     omega_t=-(lambda_0*s_dt/alpha)*np.cos(omega_0*t)
#     return omega_t
H=[H_0,[sigmay(),Omega_I]]
result_1=mesolve(H,psi_0,t_range,[],[])
phi_0=result_1.states[-1]
result=mesolve(H,phi_0,t_range,[]
               ,[rho_0,rho_1])

P_0=result.expect[0]
P_1=result.expect[1]
plt.figure()
plt.plot(t_range,P_0,label=r'$P_0$')
plt.plot(t_range,P_1,label=r'$P_1$')
plt.legend()   






















endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')