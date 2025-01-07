# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 18:39:06 2022

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

e=1.6021766208e-19
h=6.626070154e-34
h_bar=h/2/np.pi
C_c=40e-15 #fF
EC_c=e**2/(2*C_c)/h_bar/1e9
EJ_c=25*2*np.pi #7GHZ


C_q1=90e-15
C_q2=90e-15
EC_q1=e**2/(2*C_q1)/h_bar/1e9
EC_q2=e**2/(2*C_q2)/h_bar/1e9
EJ_q1=12*2*np.pi
EJ_q2=19*2*np.pi
g_qc1=200e-3*2*np.pi
g_qc2=200e-3*2*np.pi
# phi_e=0.5

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

n_zpf_c=(EJ_c/(32*EC_c))**0.25
phi_zpf_c=(2*EC_c/EJ_c)**0.25
n_c=-n_zpf_c*1j*(a-a_d)
phi_c=phi_zpf_c*(a+a_d)

H_q1=4*EC_q1*n_q1**2-EJ_q1*phi_q1.cosm()
H_q2=4*EC_q2*n_q2**2-EJ_q2*phi_q2.cosm()
H_c=4*EC_c*n_c**2-EJ_c*phi_c.cosm()

H_0=tensor(H_q1,qeye(N),qeye(N))+tensor(qeye(N),qeye(N),H_q2)+tensor(qeye(N),H_c,qeye(N))
H_g=g_qc1*tensor((a-a_d),a-a_d,qeye(N))+g_qc2*tensor(qeye(N),a-a_d,a-a_d)
#%%
num_t=401
# num_phi=101
# Phi_er=np.linspace(0,1,num_phi)

time_r=np.linspace(0,100,num_t)
[E, E_state]=H_0.eigenstates()
[Eenergy_q1,Estate_q1]=H_q1.eigenstates()
[Eenergy_q2,Estate_q2]=H_q2.eigenstates()
[Eenergy_c,Estate_c]=H_c.eigenstates()
# E=H_t.eigenenergies()
psi_11=tensor(Estate_q1[1],Estate_c[0],Estate_q2[1])
psi_20=tensor(Estate_q1[2],Estate_c[0],Estate_q2[0])
psi_02=tensor(Estate_q1[0],Estate_c[0],Estate_q2[2])
rho_11=psi_11*psi_11.dag()
rho_02=psi_02*psi_02.dag()
rho_20=psi_20*psi_20.dag()
omega20_d=(E[3]-E[5])/2/np.pi
omega02_d=(E[6]-E[5])/2/np.pi

#%%

num_t=401

time_r=np.linspace(0,100,num_t)
def g_t(t,args):
    omega_d=args['omega_d']
    return(np.sin(omega_d*2*np.pi*t))
omega20_dr=np.linspace(omega20_d*0.4,omega20_d*0.6,41)
omega02_dr=np.linspace(omega02_d*0.4,omega02_d*0.6,41)

def multy_job1(k):
    omega_d=omega02_dr[k]
    args={'omega_d':omega_d}
    H_t=[H_0,[H_g,g_t]]
    
    result=mesolve(H_t,psi_11,time_r,[],[rho_02],args=args)
    # F_list=psi_11
    F_list=result.expect[0]
    return(F_list)
# temp_folder=r'C:\Users\馒头你个史剃磅\Desktop\temp_folder'
data_1 = Parallel(n_jobs=3,mmap_mode='r+',verbose=4)(delayed(multy_job1)(k) for k in range(len(omega02_dr)))
def multy_job2(k):
    omega_d=omega20_dr[k]
    args={'omega_d':omega_d}
    H_t=[H_0,[H_g,g_t]]
    
    result=mesolve(H_t,psi_11,time_r,[],[rho_20],args=args)
    # F_list=psi_11
    F_list=result.expect[0]
    return(F_list)
# temp_folder=r'C:\Users\馒头你个史剃磅\Desktop\temp_folder'
data_2 = Parallel(n_jobs=3,mmap_mode='r+',verbose=4)(delayed(multy_job2)(k) for k in range(len(omega20_dr)))

plt.figure()
X,Y=np.meshgrid(time_r,omega02_dr/omega02_d)
plt.pcolor(X,Y,data_1,cmap='seismic')
# plt.clim(-10e-6*2*np.pi,10e-6*2*np.pi)
plt.colorbar()
plt.ylabel(r'Frequecy/$\Delta_{02}$')
plt.xlabel('time ns')
plt.figure()
X,Y=np.meshgrid(time_r,omega20_dr/omega20_d)
plt.pcolor(X,Y,data_2,cmap='seismic')
plt.ylabel(r'Frequecy/$\Delta_{20}$')
plt.xlabel('time ns')
# plt.clim(-10e-6*2*np.pi,10e-6*2*np.pi)
plt.colorbar()





