# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 20:13:06 2022

@author: mantoutou
"""
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import optimize
from joblib import Parallel, delayed
starttime=int(time.time())

N=12
# N_c=11
e=1.6021766208e-19
h=6.626070154e-34
h_bar=h/2/np.pi

C_q1=80e-15
C_q2=80e-15
C_c=40e-15 #fF
C_12=0.8e-15 #fF
C_g1=7.9e-15
C_g2=7.9e-15

C_t3=(C_q1+C_g1+C_12)*(C_q2+C_g2+C_12)*(C_c+C_g1+C_g2)-2*C_g1*C_g2*C_12\
    -C_12**2*(C_c+C_g1+C_g2)-C_g2**2*(C_q1+C_g1+C_12)-C_g1**2*(C_q2+C_g2+C_12)
Cq1_p=C_t3/((C_q2+C_g2+C_12)*(C_c+C_g1+C_g2)-C_g2**2)
Cq2_p=C_t3/((C_q1+C_g1+C_12)*(C_c+C_g1+C_g2)-C_g2**2)
Cc_p=C_t3/((C_q1+C_g1+C_12)*(C_q2+C_g2+C_12)-C_12**2)
Cg1_p=C_t3/(C_g1*(C_q2+C_g2+C_12)+C_g2*C_12)
Cg2_p=C_t3/(C_g2*(C_q1+C_g1+C_12)+C_g1*C_12)
C12_p=C_t3/(C_g1*C_g2+(C_c+C_g1+C_g2)*C_12)

EC_q1=e**2/(2*Cq1_p)/h_bar/1e9
EC_q2=e**2/(2*Cq2_p)/h_bar/1e9
EC_c=e**2/(2*Cc_p)/h_bar/1e9
EC_g1=e**2/(Cg1_p)/h_bar/1e9
EC_g2=e**2/(Cg2_p)/h_bar/1e9
EC_12=e**2/(C12_p)/h_bar/1e9

EJ_c=26*2*np.pi
EJ_q1=12.5*2*np.pi
EJ_q2=12.5*2*np.pi
# EJ_c_list=np.linspace(12,22,51)*2*np.pi

"""""""""比特哈密顿"""""""""
H_cphi=1/2*(qdiags([1]*(N-1),1)+qdiags([1]*(N-1),-1))
# H_sphi=-1j/2*(qdiags([1]*(N_c-1),1)-qdiags([1]*(N_c-1),-1))
n_zpf_q1=(EJ_q1/(32*EC_q1))**0.25
n_zpf_q2=(EJ_q2/(32*EC_q2))**0.25
n_zpf_c=(EJ_c/(32*EC_c))**0.25
# phi_zpf=(2*E_C/E_J)**0.25
g = 4*EC_g1/n_zpf_q1/n_zpf_c
n_range=np.linspace(0,N,N+1)
def H_n():
    H_t=0
    for i in range(N):
        n=int(n_range[i])
        H_c=(n-int(N/2))*basis(N,n)*basis(N,n).dag()
        H_t=H_t+H_c
    return(H_t)

H_q1=4*EC_q1*H_n()**2-EJ_q1*H_cphi
H_q2=4*EC_q2*H_n()**2-EJ_q2*H_cphi
H_c=4*EC_c*H_n()**2-EJ_c*H_cphi

H_q1t=tensor(H_q1,qeye(N),qeye(N))
H_q2t=tensor(qeye(N),qeye(N),H_q2)
H_ct=tensor(qeye(N),H_c,qeye(N))
H_gt=4*EC_g1*tensor(H_n(),H_n(),qeye(N))+4*EC_g2*tensor(qeye(N),H_n(),H_n())\
    +4*EC_12*tensor(H_n(),qeye(N),H_n())
H_t=H_q1t+H_q2t+H_ct+H_gt
    
num_t=301
time_r=np.linspace(0,100,num_t)
[Eenergy_q1,Estate_q1]=H_q1.eigenstates()
[Eenergy_q2,Estate_q2]=H_q2.eigenstates()
[Eenergy_c,Estate_c]=H_c.eigenstates()
[E_energy,E_state]=H_t.eigenstates()

psi_100=tensor(Estate_q1[1],Estate_c[0],Estate_q2[0]).unit()
psi_001=tensor(Estate_q1[0],Estate_c[0],Estate_q2[1]).unit()
psi_010=tensor(Estate_q1[0],Estate_c[1],Estate_q2[0]).unit()
psi_001_T = E_state[1]
rho_100=psi_100*psi_100.dag()
rho_001=psi_001*psi_001.dag()
#%%
result=mesolve(H_t,psi_100,time_r,[],[rho_100])

F_list=result.expect[0]
# # for i in range(num_t):
# #     print(i)
# #     psi_i=result.states[i]
# #     rho_i=psi_i*psi_i.dag()
# #     F_i=(rho_i*rho_10).tr()
# #     F_list.append(F_i)
plt.figure()
plt.plot(time_r,F_list)
endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')
#%%
EJ_c_list=np.linspace(14,26,20)*2*np.pi
def multy_job(k):
    EJ_c=EJ_c_list[k]
    H_c=4*EC_c*H_n()**2-EJ_c*H_cphi
    H_ct=tensor(qeye(N),H_c,qeye(N))
    H_t=H_q1t+H_q2t+H_ct+H_gt
    
    [Eenergy_q1,Estate_q1]=H_q1.eigenstates()
    [Eenergy_q2,Estate_q2]=H_q2.eigenstates()
    [Eenergy_c,Estate_c]=H_c.eigenstates()
    omega_c=(Eenergy_c[1]-Eenergy_c[0])/2/np.pi
    psi_10=tensor(Estate_q1[1],Estate_c[0],Estate_q2[0])
    psi_01=tensor(Estate_q1[0],Estate_c[0],Estate_q2[1])
    rho_10=psi_10*psi_10.dag()
    rho_01=psi_01*psi_01.dag()
    
    result=mesolve(H_t,psi_01,time_r,[],[rho_10])
    
    F_list=result.expect[0]
    return(F_list,omega_c)
data = Parallel(n_jobs=10, verbose=2)(delayed(multy_job)(k) for k in range(len(EJ_c_list)))

data_f=[data[i][0] for i in range(len(EJ_c_list))]
data_omegac=[data[i][1] for i in range(len(EJ_c_list))]
plt.figure()
X,Y=np.meshgrid(time_r,data_omegac)
plt.pcolor(X,Y,data_f,cmap='seismic')
# plt.clim(-10e-6*2*np.pi,10e-6*2*np.pi)
plt.colorbar()

