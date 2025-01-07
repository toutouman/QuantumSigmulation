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

N=8
E_C=0.3176*2*np.pi  #GHZ
E_J=14*2*np.pi #GHZ
num=201
sigma_0=3
g=1/np.sqrt(2*np.pi)/sigma_0*1.019 #0.0125*2*np.pi,1.019来自Q_zpf的偏差

T=24 #18结果更好
t_range=np.linspace(0,T,num) #ns
N_g=np.linspace(-2,2,num)
n_range=np.linspace(0,N,N+1)
n_g=0.5

a_d=create(N)
a=destroy(N)
# Omega=(E_J/E_C/8)**0.25/np.sqrt(2)
# lambda_0=0.5/2.976
# t_0=T/2

n_zpf=(E_J/(32*E_C))**0.25
phi_zpf=(2*E_C/E_J)**0.25
n=-n_zpf*1j*(a-a_d)
phi=phi_zpf*(a+a_d)
H_t=4*E_C*n**2+E_J*phi**2/2-E_J/24*phi**4+E_J/720*phi**6-E_J/40320*phi**8+E_J/3628800*phi**10
# H_t=4*E_C*n**2-E_J*phi.cosm()
# H_t=np.sqrt(8*E_J*E_C)*(a_d*a)-(a+a_d)**4*E_C/12+(a+a_d)**6*E_C/360*(2*E_C/E_J)**0.5

# Q_zpf=np.sqrt(1/(2*np.sqrt(8*E_J*E_C)))
Q_zpf=(E_J/E_C/8)**0.25/np.sqrt(2)
H_n=-1j*Q_zpf*(a-a_d)


def Omega_I(t,arg):
    phi_0=arg['phi_0']
    omega_xy=arg['omega_xy']
    s_t=g/Q_zpf*np.exp(-(t-t_0)**2/(2*sigma_0**2))*np.pi/1.36895/2
    omega_t=s_t*np.sin(omega_xy*(t+phi_0))
    return omega_t
def Omega_Q(t,arg):
    phi_0=arg['phi_0']
    omega_xy=arg['omega_xy']
    s_dt=-g/Q_zpf*np.exp(-(t-t_0)**2/(2*sigma_0**2))*(t-t_0)/sigma_0**2*np.pi/1.36895/2
    omega_t=-(lambda_0*s_dt/alpha)*np.cos(omega_xy*(t+phi_0))
    return omega_t


E_0=H_t.eigenenergies()[0]
E_1=H_t.eigenenergies()[1]
E_2=H_t.eigenenergies()[2]
psi_0=H_t.eigenstates()[1][0]
psi_1=H_t.eigenstates()[1][1]
psi_2=H_t.eigenstates()[1][2]
rho_0=psi_0*psi_0.dag()
rho_1=psi_1*psi_1.dag()
rho_2=psi_2*psi_2.dag()
omega_0=E_1-E_0
omega_1=E_2-E_1
omega_02=(omega_0+omega_1)
alpha=omega_0-omega_1
print(omega_0/2/np.pi)
#%%
H_xy=[H_t,[H_n,Omega_I],[H_n,Omega_Q]]

arg_1={'phi_0':0,'omega_xy':omega_1}
result_1=mesolve(H_xy,psi_1,t_range,[] ,[],args=arg_1)
psi_t1=result_1.states[-1]


arg_2={'phi_0':T,'omega_xy':omega_1}
a_d=create(N)
a=destroy(N)
Gamma=1/3/1e3
T_range=np.linspace(0,250*T,250*num)
result_2=mesolve(H_t,psi_t1,T_range,[np.sqrt(Gamma)*a],[],
                args=arg_2)
rho_t3=result_2.states[-1]

arg_3={'phi_0':151*T,'omega_xy':omega_1}
result=mesolve(H_xy,rho_t3,t_range,[],[rho_0,rho_1,rho_2],
                args=arg_3)


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

time_r=np.linspace(0,50,1001)

plt.figure()
plt.plot(time_r,[T2_delay_22(i,d_omega) for i in time_r],label='P_22')
plt.plot(time_r,[T2_delay_11(i,d_omega) for i in time_r],label='P_11')
plt.plot(time_r,[1-T2_delay_11(i,d_omega)-T2_delay_22(i,d_omega) for i in time_r],label='P_00')
plt.legend()
    

endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')