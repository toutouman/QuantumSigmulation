# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 20:44:00 2022

@author: 馒头你个史剃磅
"""



from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import optimize
from joblib import Parallel, delayed
starttime=int(time.time())

N=8
N_c=13
e=1.6021766208e-19
h=6.626070154e-34
h_bar=h/2/np.pi
C_c1=40e-15 #fF
C_c2=40e-15 #fF
C_12=2e-15 #fF
C_t=np.sqrt(C_c1*C_c2+C_c1*C_12+C_c2*C_12)
EC_c1=(C_c2+C_12)*e**2/(2*C_t**2)/h_bar/1e9
EC_c2=(C_c1+C_12)*e**2/(2*C_t**2)/h_bar/1e9
EC_12=(C_12)*e**2/(C_t**2)/h_bar/1e9
EJ_c1=22*2*np.pi #GHZ
EJ_c2=22*2*np.pi
EJ_12=(EJ_c1)/4

EJ_c2_list=np.linspace(24,44,41)*2*np.pi

C_q1=100e-15
C_q2=100e-15
EC_q1=e**2/(2*C_q1)/h_bar/1e9
EC_q2=e**2/(2*C_q2)/h_bar/1e9
EJ_q1=12*2*np.pi
EJ_q2=12*2*np.pi
g_qc1=200e-3*2*np.pi
g_qc2=200e-3*2*np.pi
phi_e12=0.5

num_t=251
num_phi=101
Phi_er=np.linspace(0,0.5,num_phi)


# Phi_er=[Phie12toe(i) for i in Phi_e12r]

time_r=np.linspace(0,100,num_t)

a=destroy(N)
a_d=create(N)


H_cphi=1/2*(qdiags([1]*(N_c-1),1)+qdiags([1]*(N_c-1),-1))
H_sphi=1j/2*(qdiags([1]*(N_c-1),1)-qdiags([1]*(N_c-1),-1))

n_range=np.linspace(0,N_c,N_c+1)
def H_n():
    H_t=0
    for i in range(N_c):
        n=int(n_range[i])
        H_c=(n-int(N_c/2))*basis(N_c,n)*basis(N_c,n).dag()
        H_t=H_t+H_c
    return(H_t)

EJ_c1p=EJ_c1+EJ_12
n_zpf_c1=(EJ_c1p/(32*EC_c1))**0.25
Hn_c1=tensor(H_n(),qeye(N))
H_cphi1=tensor(H_cphi,qeye(N))
H_sphi1=tensor(H_sphi,qeye(N))
# n_c1=tensor(H_n(),qeye(N))

EJ_c2p=EJ_c2+EJ_12
n_zpf_c2=(EJ_c2p/(32*EC_c2))**0.25
phi_zpf_c2=(2*EC_c2/EJ_c2p)**0.25
Hn_c2=tensor(qeye(N_c),-n_zpf_c2*1j*(a-a_d))
phi_c2=tensor(qeye(N_c),phi_zpf_c2*(a+a_d))
H_cphi2=phi_c2.cosm()
H_sphi2=phi_c2.sinm()
def H_ct(phi_ei):
    phi_e=phi_ei*2*np.pi
    
    H_C=4*EC_c1*Hn_c1**2+4*EC_c2*Hn_c2**2+4*EC_12*Hn_c1*Hn_c2
    # H_phi12=EJ_12p*dphi_e12/EJ_c1p*(C_c1*4*EC_c1*n_c1/e+C_c1*2*EC_12/e*n_c2)-\
    #     EJ_12p*dphi_e12/EJ_c2p*(C_c2*4*EC_c2*n_c2/e+C_c2*2*EC_12/e*n_c1)
    H_J=-EJ_c1*(H_cphi1*np.cos(phi_e)-H_sphi1*np.sin(phi_e))-EJ_c2*H_cphi2-EJ_12*(H_cphi1*H_cphi2+H_sphi1*H_sphi2)
    H=H_C+H_J
    H_tensor=tensor(qeye(N),H,qeye(N))
    return({'H':H, 'tensor_H':H_tensor})

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
H_q1t=tensor(H_q1,qeye(N_c),qeye(N),qeye(N))
H_q2t=tensor(qeye(N),qeye(N_c),qeye(N),H_q2)
H_qc1=-1j*g_qc1*tensor((a-a_d),H_n(),qeye(N),qeye(N))/n_zpf_c1
H_qc2=-g_qc2*tensor(qeye(N),qeye(N_c),(a-a_d),(a-a_d))


phi_e=0
H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e)['tensor_H']

[Eenergy_q1,Estate_q1]=H_q1.eigenstates()
[Eenergy_q2,Estate_q2]=H_q2.eigenstates()
[Eenergy_c,Estate_c]=H_ct(phi_e)['H'].eigenstates()
# [E_energy,E_state]=H_t.eigenstates()

psi_10=tensor(Estate_q1[1],Estate_c[0],Estate_q2[0])
psi_01=tensor(Estate_q1[0],Estate_c[0],Estate_q2[1])
# psi_00=E_state[0]
# psi_10=E_state[1]
# psi_01=E_state[2]
rho_10=psi_10*psi_10.dag()
# rho_10=tensor(Estate_q1[1]*Estate_q1[1].dag(),qeye(N),qeye(N),Estate_q2[0]*Estate_q2[0].dag())
rho_01=psi_01*psi_01.dag()

result=mesolve(H_t,psi_01,time_r,[],[rho_10])

F_list=result.expect[0]
# for i in range(num_t):
#     psi_i=result.states[i]
#     rho_i=psi_i*psi_i.dag()
#     F_i=(rho_i*rho_10).tr()
#     F_list.append(F_i)
plt.figure()
plt.plot(time_r,F_list)



endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')
#%%    
def multy_job(k):
    phi_e=Phi_er[k]
    H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e)['tensor_H']
    
    [Eenergy_q1,Estate_q1]=H_q1.eigenstates()
    [Eenergy_q2,Estate_q2]=H_q2.eigenstates()
    [Eenergy_c,Estate_c]=H_ct(phi_e)['H'].eigenstates()
    [E_energy,E_state]=H_t.eigenstates()
    
    psi_10=tensor(Estate_q1[1],Estate_c[0],Estate_q2[0])
    psi_01=tensor(Estate_q1[0],Estate_c[0],Estate_q2[1])
    # psi_00=E_state[0]
    # psi_10=E_state[1]
    # psi_01=E_state[2]
    rho_10=psi_10*psi_10.dag()
    # rho_10=tensor(Estate_q1[1]*Estate_q1[1].dag(),qeye(N),qeye(N),Estate_q2[1]*Estate_q2[1].dag())
    rho_01=psi_01*psi_01.dag()
    
    result=mesolve(H_t,psi_01,time_r,[],[rho_10])
    F_list=result.expect[0]
    # F_list=[]
    # for i in range(num_t):
    #     psi_i=result.states[i]
    #     rho_i=psi_i*psi_i.dag()
    #     F_i=(rho_i*rho_10).tr()
    #     F_list.append(F_i)
    return(F_list)
data = Parallel(n_jobs=7, verbose=2)(delayed(multy_job)(k) for k in range(len(Phi_er)))
plt.figure()
X,Y=np.meshgrid(time_r,Phi_e12r)
plt.pcolor(Y,X,np.real(data),cmap='jet')
plt.clim(0,1)
plt.colorbar()

def fit_geff(t,g,A,B,phi):
    func=np.cos(2*np.pi*g*t+phi)*A+B
    return(func)
Geff_list=[]
bounds=((0,0.4,0.4,0),(1,0.6,0.6,2*np.pi))
for i in range(len(Phi_e12r)):
    index_i=np.where(np.real(data[i])>0.95)[0][0] if np.max(data[i])>0.95 else -1
    g_0=0.5/time_r[index_i]
    g_i,A_i,B_i,phi_i=optimize.curve_fit(fit_geff, time_r, np.real(data[i]),[g_0,0.5,0.5,np.pi],bounds=bounds,maxfev = 40000)[0]
    Geff_list.append(g_i*1e3)
    # plt.figure()
    # plt.plot(time_r, np.real(data[i]))
    # plt.plot(time_r, [fit_geff(i,g_i,A_i,B_i,phi_i) for i in time_r])
# plt.figure()
plt.plot(Phi_e12r,np.abs(Geff_list),label=r'$C_{12}$=0 fF')
plt.plot(Phi_e12r,np.abs(Geff_list1),label=r'$C_{12}$=2 fF')
plt.ylabel(r'$2g_{eff} $ (MHz)')
plt.xlabel(r'$\phi_e/\phi_0$')
plt.legend()
 #%%
Omega_0=[]
Omega_1=[]
Omega_2=[]
Omega_3=[]
Omega_4=[]
for i in range(num_phi):
    phi_e=Phi_er[i]
    # E_energy=H_ct(phi_e).eigenenergies()
    [E_energy,E_state]=H_ct(phi_e)['H'].eigenstates()
    E_0=E_energy[0]
    E_1=E_energy[1]
    E_2=E_energy[2]
    E_3=E_energy[3]
    E_4=E_energy[4]
    E_5=E_energy[5]
    psi_0=E_state[0]
    psi_1=E_state[1]
    psi_2=E_state[2]
    rho_0=psi_0*psi_0.dag()
    rho_1=psi_1*psi_1.dag()
    rho_2=psi_2*psi_2.dag()
    Omega_0.append((E_1-E_0)/2/np.pi)
    Omega_1.append((E_2-E_0)/2/np.pi)
    Omega_2.append((E_3-E_0)/2/np.pi)
    Omega_3.append((E_4-E_0)/2/np.pi)
    Omega_4.append((E_5-E_0)/2/np.pi)
    # omega_02=(omega_0+omega_1)/2
    # alpha=omega_0-omega_1

plt.figure()
plt.plot(Phi_er,Omega_0)
plt.plot(Phi_er,Omega_1)
plt.plot(Phi_er,Omega_2)
plt.plot(Phi_er,Omega_3)
# plt.plot(Phi_er,Omega_4)
# plt.ylim([0,10])
endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')
