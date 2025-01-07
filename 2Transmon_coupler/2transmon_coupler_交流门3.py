# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 19:49:33 2022

@author: mantoutou
"""


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import optimize
from joblib import Parallel, delayed
starttime=int(time.time())

N=8
N_c=12
e=1.6021766208e-19
h=6.626070154e-34
h_bar=h/2/np.pi
C_c1=60e-15 #fF
C_c2=60e-15 #fF
C_12=12e-15 #fF
C_t=np.sqrt(C_c1*C_c2+C_c1*C_12+C_c2*C_12)
EC_c1=(C_c2+C_12)*e**2/(2*C_t**2)/h_bar/1e9
EC_c2=(C_c1+C_12)*e**2/(2*C_t**2)/h_bar/1e9
EC_12=(C_12)*e**2/(C_t**2)/h_bar/1e9
EJ_c1=22*2*np.pi #7GHZ
EJ_c2=22*2*np.pi #8.5GHZ
EJ_12=15*2*np.pi


# EJ_c2_list=np.linspace(24,44,41)*2*np.pi

EJ_c2_list=np.linspace(20,24,41)*2*np.pi

C_q1=90e-15
C_q2=90e-15
EC_q1=e**2/(2*C_q1)/h_bar/1e9
EC_q2=e**2/(2*C_q2)/h_bar/1e9
EJ_q1=13*2*np.pi
EJ_q2=18*2*np.pi
g_qc1=200e-3*2*np.pi
g_qc2=200e-3*2*np.pi
# phi_e=0.5
# phi_e=0.5
# phi_e0=0

num_t=401
num_phi=101
Phi_er=np.linspace(0,1,num_phi)

time_r=np.linspace(0,100,num_t)
#%%
a=destroy(N)
a_d=create(N)

n_zpf_q1=(EJ_q1/(32*EC_q1))**0.25
phi_zpf_q1=(2*EC_q1/EJ_q1)**0.25
n_q1=-n_zpf_q1*1j*(a-a_d)
phi_q1=phi_zpf_q1*(a+a_d)
n_zpf_q2=(EJ_q2/(32*EC_q2))**0.25
phi_zpf_q2=(2*EC_q2/EJ_q2)**0.25
n_q2=-n_zpf_q2*1j*(a-a_d)
phi_q2=phi_zpf_q2*(a+a_d)

H_q1=4*EC_q1*n_q1**2-EJ_q1*phi_q1.cosm()
H_q2=4*EC_q2*n_q2**2-EJ_q2*phi_q2.cosm()


H_cphi=1/2*(qdiags([1]*(N_c-1),1)+qdiags([1]*(N_c-1),-1))
H_sphi=-1j/2*(qdiags([1]*(N_c-1),1)-qdiags([1]*(N_c-1),-1))

n_range=np.linspace(0,N_c,N_c+1)
def H_n():
    H_t=0
    for i in range(N_c):
        n=int(n_range[i])
        H_c=(n-int(N_c/2))*basis(N_c,n)*basis(N_c,n).dag()
        H_t=H_t+H_c
    return(H_t)

Hn_c1=tensor(H_n(),qeye(N_c))
Hn_c2=tensor(qeye(N_c),H_n())
H_cphi1=tensor(H_cphi,qeye(N_c))
H_cphi2=tensor(qeye(N_c),H_cphi)
H_sphi1=tensor(H_sphi,qeye(N_c))
H_sphi2=tensor(qeye(N_c),H_sphi)
    #%%

H_C0=4*EC_c1*Hn_c1**2+4*EC_c2*Hn_c2**2+4*EC_12*Hn_c1*Hn_c2
H_C0t=tensor(qeye(N),H_C0,qeye(N))

H_J1c=tensor(qeye(N),-EJ_c1*H_cphi1,qeye(N))
H_J1s=tensor(qeye(N),+EJ_c1*H_sphi1,qeye(N))

H_J2c=tensor(qeye(N),-EJ_c2*H_cphi2,qeye(N))
H_J2s=tensor(qeye(N),-EJ_c2*H_sphi2,qeye(N))

H_J12c=tensor(qeye(N),-EJ_12*(H_cphi1*H_cphi2+H_sphi1*H_sphi2),qeye(N))
H_J12s=tensor(qeye(N),EJ_12*(H_sphi2*H_cphi1-H_cphi2*H_sphi1),qeye(N))
# H_dphi=tensor(qeye(N),(Hn_c1*(C_c2*C_12)-Hn_c2*C_c1*C_12)/C_t**2,qeye(N))

def H_ct(phi_ei):
    phi_e=phi_ei*2*np.pi
    H_C=4*EC_c1*Hn_c1**2+4*EC_c2*Hn_c2**2+4*EC_12*Hn_c1*Hn_c2
    # H_J1=-EJ_c1*(np.exp(1j*C_12*C_c2/C_t**2*phi_e)*H_cphi1)
    # H_J2=-EJ_c2*(np.exp(-1j*C_12*C_c1/C_t**2*phi_e)*H_cphi2)
    # H_dphi=-dphi*(-2*e*Hn_c1*C_c2*C_c12+2*e*Hn_c2*C_c1*C_12)/C_t**2/hbar
    H_J1=-EJ_c1*(H_cphi1*np.cos(C_12*C_c2/C_t**2*phi_e)-H_sphi1*np.sin(C_12*C_c2/C_t**2*phi_e))
    H_J2=-EJ_c2*(H_cphi2*np.cos(C_12*C_c1/C_t**2*phi_e)+H_sphi2*np.sin(C_12*C_c1/C_t**2*phi_e))
    H_J12=-EJ_12*(H_cphi1*H_cphi2+H_sphi1*H_sphi2)*np.cos(C_c1*C_c2/C_t**2*phi_e)+\
        EJ_12*(H_sphi2*H_cphi1-H_cphi2*H_sphi1)*np.sin(C_c1*C_c2/C_t**2*phi_e)

    H=H_C+H_J1+H_J2+H_J12
    H_tensor=tensor(qeye(N),H,qeye(N))
    return({'H':H, 'tensor_H':H_tensor})
EJ_c1p=(EJ_c1+EJ_12)
EJ_c2p=(EJ_c2+EJ_12)
n_zpf_c1=(EJ_c1p/(32*EC_c1))**0.25
n_zpf_c2=(EJ_c2p/(32*EC_c2))**0.25
H_q1t=tensor(H_q1,qeye(N_c),qeye(N_c),qeye(N))
H_q2t=tensor(qeye(N),qeye(N_c),qeye(N_c),H_q2)
H_qc1=-1j*g_qc1*tensor((a-a_d),H_n(),qeye(N_c),qeye(N))/n_zpf_c1
H_qc2=-1j*g_qc2*tensor(qeye(N),qeye(N_c),H_n(),(a-a_d))/n_zpf_c2
# H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e0)['tensor_H']



#%%
num_t=401
# num_phi=101
# Phi_er=np.linspace(0,1,num_phi)

time_r=np.linspace(0,100,num_t)
phi_e0=0.35
H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e0)['tensor_H']
[E, E_state]=H_t.eigenstates()
psi_11=E_state[6]
psi_20=E_state[5]
psi_02=E_state[7]
rho_11=psi_11*psi_11.dag()
rho_02=psi_02*psi_02.dag()
rho_20=psi_20*psi_20.dag()
# rho_01=psi_01*psi_01.dag()
omega20_d=(E[5]-E[6])/2/np.pi/2
omega02_d=(E[7]-E[6])/2/np.pi/2
#%%
A=0.085
omega20_dr=np.linspace(omega20_d*1.95,omega20_d*2.02,11)
omega02_dr=np.linspace(omega02_d*1.97,omega02_d*2.02,6)
print(phi_e0)
# def cos_phie(t,args):
#     omega_d=args['omega_d']
#     phi_e=A*np.sin(omega_d*2*np.pi*t)+phi_e0
#     return(np.cos(phi_e*2*np.pi))


# def sin_phie(t,args):
#     omega_d=args['omega_d']
#     phi_e=A*np.sin(omega_d*2*np.pi*t)+phi_e0
#     return(np.sin(phi_e*2*np.pi))
# def d_phie(t,args):
#     omega_d=args['omega_d']
#     d_phie=(omega_d*2*np.pi)*A*np.cos(omega_d*2*np.pi*t)*2*np.pi
#     return(d_phie)
def cos_phie1(t,args):
    omega_d=args['omega_d']
    phi_e=A*np.sin(omega_d*2*np.pi*t)+phi_e0
    return(np.cos(C_12*C_c2/C_t**2*phi_e*2*np.pi))

def sin_phie1(t,args):
    omega_d=args['omega_d']
    phi_e=A*np.sin(omega_d*2*np.pi*t)+phi_e0
    return(np.sin(C_12*C_c2/C_t**2*phi_e*2*np.pi))

def cos_phie2(t,args):
    omega_d=args['omega_d']
    phi_e=A*np.sin(omega_d*2*np.pi*t)+phi_e0
    return(np.cos(C_12*C_c1/C_t**2*phi_e*2*np.pi))
def sin_phie2(t,args):
    omega_d=args['omega_d']
    phi_e=A*np.sin(omega_d*2*np.pi*t)+phi_e0
    return(np.sin(C_12*C_c1/C_t**2*phi_e*2*np.pi))

def cos_phie12(t,args):
    omega_d=args['omega_d']
    phi_e=A*np.sin(omega_d*2*np.pi*t)+phi_e0
    return(np.cos(C_c2*C_c1/C_t**2*phi_e*2*np.pi))
def sin_phie12(t,args):
    omega_d=args['omega_d']
    phi_e=A*np.sin(omega_d*2*np.pi*t)+phi_e0
    return(np.sin(C_c2*C_c1/C_t**2*phi_e*2*np.pi))

    
def multy_job1(k):
    omega_d=omega02_dr[k]
    args={'omega_d':omega_d}
    H_t=[H_C0t+H_q1t+H_q2t+H_qc1+H_qc2,
         [H_J1c,cos_phie1],[H_J1s,sin_phie1],
         [H_J2c,cos_phie2],[H_J2s,sin_phie2],
         [H_J12c,cos_phie12],[H_J12s,sin_phie12]]
    
    result=mesolve(H_t,psi_11,time_r,[],[rho_02],args=args)
    # F_list=psi_11
    F_list=result.expect[0]
    return(F_list)
temp_folder=r'C:\Users\馒头你个史剃磅\Desktop\temp_folder'
data_1 = Parallel(n_jobs=2,mmap_mode='r+',temp_folder=temp_folder,verbose=4)(delayed(multy_job1)(k) for k in range(len(omega02_dr)))
# Data1_list.append(data_1)
def multy_job2(k):
    omega_d=omega20_dr[k]
    args={'omega_d':omega_d}
    H_t=[H_C0t+H_q1t+H_q2t+H_qc1+H_qc2,
         [H_J1c,cos_phie1],[H_J1s,sin_phie1],
         [H_J2c,cos_phie2],[H_J2s,sin_phie2],
         [H_J12c,cos_phie12],[H_J12s,sin_phie12]]
    
    result=mesolve(H_t,psi_11,time_r,[],[rho_20],args=args)
    # F_list=psi_11
    F_list=result.expect[0]
    return(F_list)
temp_folder=r'C:\Users\mantoutou\Desktop\temp_folder'
data_2 = Parallel(n_jobs=8,mmap_mode='r+',temp_folder=temp_folder,verbose=4)(delayed(multy_job2)(k) for k in range(len(omega20_dr)))
# Data2_list.append(data_2)


plt.figure()
X,Y=np.meshgrid(time_r,omega02_dr/omega02_d)
plt.pcolor(X,Y,data_1,cmap='seismic')
# plt.clim(0,1)
plt.colorbar()
plt.ylabel(r'Frequecy/$0.5\omega_{02}$')
plt.xlabel('time ns')

plt.figure()
X,Y=np.meshgrid(time_r,omega20_dr/omega20_d)
plt.pcolor(X,Y,data_2,cmap='seismic')
plt.ylabel(r'Frequecy/$0.5\omega_{20}$')
plt.xlabel('time ns')
plt.clim(0,1)
# plt.clim(-10e-6*2*np.pi,10e-6*2*np.pi)
plt.colorbar()

plt.figure()
plt.plot(time_r,data_1[-1])
plt.ylabel(r'$|\langle02|11\rangle|^2$')
plt.xlabel('time ns')
