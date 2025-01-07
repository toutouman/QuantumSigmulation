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

N=9

e=1.6021766208e-19
h=6.626070154e-34
h_bar=h/2/np.pi
C_c1=70e-15 #fF
C_c2=70e-15 #fF
C_12=30e-15 #fF
C_t=np.sqrt(C_c1*C_c2+C_c1*C_12+C_c2*C_12)
EC_c1=(C_c2+C_12)*e**2/(2*C_t**2)/h_bar/1e9
EC_c2=(C_c1+C_12)*e**2/(2*C_t**2)/h_bar/1e9
EC_12=(C_12)*e**2/(C_t**2)/h_bar/1e9
EJ_c1=41*2*np.pi #7GHZ
EJ_c2=41*2*np.pi #8.5GHZ
EJ_12=41*2*np.pi*0.2347


# EJ_c2_list=np.linspace(24,44,41)*2*np.pi

EJ_c2_list=np.linspace(20,24,41)*2*np.pi

C_q1=85e-15
C_q2=85e-15
EC_q1=e**2/(2*C_q1)/h_bar/1e9
EC_q2=e**2/(2*C_q2)/h_bar/1e9
EJ_q1=23*2*np.pi
EJ_q2=26*2*np.pi
g_qc1=149e-3*2*np.pi
g_qc2=149e-3*2*np.pi

num_t=501
time_r=np.linspace(0,100,num_t)

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

def H_ct(phi_ei):
    phi_e=phi_ei*2*np.pi
    EJ_c1p=EJ_c1+EJ_12*np.cos(phi_e)
    EJ_c2p=EJ_c2+EJ_12*np.cos(phi_e)
    # EJ_12p=EJ_12*np.cos(phi_e12)
    n_zpf_c1=((EJ_c1p)/(32*EC_c1))**0.25
    n_zpf_c2=((EJ_c2p)/(32*EC_c2))**0.25
    phi_zpf_c1=(2*EC_c1/(EJ_c1p))**0.25
    phi_zpf_c2=(2*EC_c2/(EJ_c2p))**0.25
    n_c1=tensor(-n_zpf_c1*1j*(a-a_d),qeye(N))
    n_c2=tensor(qeye(N),-n_zpf_c2*1j*(a-a_d))
    phi_c1=tensor(phi_zpf_c1*(a+a_d),qeye(N))
    phi_c2=tensor(qeye(N),phi_zpf_c2*(a+a_d))
    dphi_e12=0
    
    H_C=4*EC_c1*n_c1**2+4*EC_c2*n_c2**2+4*EC_12*n_c1*n_c2
    # H_phi12=EJ_12p*dphi_e12/EJ_c1p*(C_c1*4*EC_c1*n_c1/e+C_c1*2*EC_12/e*n_c2)-\
    #     EJ_12p*dphi_e12/EJ_c2p*(C_c2*4*EC_c2*n_c2/e+C_c2*2*EC_12/e*n_c1)
    H_J=-EJ_c1*phi_c1.cosm()-EJ_c2*phi_c2.cosm()-EJ_12*(np.cos(phi_e)*(phi_c2-phi_c1).cosm()+np.sin(phi_e)*(phi_c2-phi_c1).sinm())
    H=H_C+H_J
    H_tensor=tensor(qeye(N),H,qeye(N))
    return({'H':H, 'tensor_H':H_tensor})

H_q1t=tensor(H_q1,qeye(N),qeye(N),qeye(N))
H_q2t=tensor(qeye(N),qeye(N),qeye(N),H_q2)
H_qc1=-g_qc1*tensor((a-a_d),(a-a_d),qeye(N),qeye(N))
H_qc2=-g_qc2*tensor(qeye(N),qeye(N),(a-a_d),(a-a_d))

#%% 测量ZZ耦合
num_phi=21
Phi_er=np.linspace(0.0,0.3,num_phi)
Xi_list=[]
def multy_ZZ(k):
    # print(i)
    phi_e=Phi_er[k]
    H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e)['tensor_H']
    # [E_energy,E_state]=H_t.eigenstates()
    E_energy=H_t.eigenenergies()
    # [Eenergy_q1,Estate_q1]=H_q1.eigenstates()
    # [Eenergy_q2,Estate_q2]=H_q2.eigenstates()
    # [Eenergy_c,Estate_c]=H_ct(phi_e0)['H'].eigenstates()
    xi_i=E_energy[4]+E_energy[0]-E_energy[1]-E_energy[2]
    # Xi_list.append(xi_i/2/np.pi)
    return(xi_i/2/np.pi)
    
Xi_list=Parallel(n_jobs=3,verbose=4)(delayed(multy_ZZ)(k) for k in range(num_phi))

plt.figure()
plt.plot(Phi_er,[i*1e6 for i in Xi_list],label='ZZ coupling')
plt.xlabel(r'$\phi_e$')
plt.ylabel(r'$\xi_{zz}$ kHz')
plt.plot([np.min(Phi_er),np.max(Phi_er)],[0,0],'k--')
plt.legend()
#%%测量串扰大小
phi_e0=0.25
H_t0=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e0)['tensor_H']
[E_energy,E_state]=H_t0.eigenstates()
[Eenergy_q1,Estate_q1]=H_q1.eigenstates()
[Eenergy_q2,Estate_q2]=H_q2.eigenstates()
[Eenergy_c,Estate_c]=H_ct(phi_e0)['H'].eigenstates()
psi_00=tensor(Estate_q1[0],Estate_c[0],Estate_q2[0])
psi_10=tensor(Estate_q1[1],Estate_c[0],Estate_q2[0])
psi_01=tensor(Estate_q1[0],Estate_c[0],Estate_q2[1])
psi_11=tensor(Estate_q1[1],Estate_c[0],Estate_q2[1])
rho_01=psi_01*psi_01.dag()
rho_10=psi_10*psi_10.dag()
psi_1=E_state[1]
psi_2=E_state[2]
rho_1=psi_1*psi_1.dag()
rho_2=psi_2*psi_2.dag()
p_11=(rho_10*rho_1).tr()
p_12=(rho_01*rho_1).tr()
p_21=(rho_10*rho_2).tr()
p_22=(rho_01*rho_2).tr()
print(p_11,p_12,'\n',p_22,p_21)
#%% 测量等效耦合
phi_e0=0.35
H_t0=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e0)['tensor_H']
[E_energy,E_state]=H_t0.eigenstates()
S_s=(E_state[0]).full()
for i in range(len(E_state)-1):
    S_s=np.column_stack((S_s,E_state[i+1].full()))
S_q=Qobj(S_s,dims=[[8,8,8,8],[8,8,8,8]])
# [Eenergy_q1,Estate_q1]=H_q1.eigenstates()
# [Eenergy_q2,Estate_q2]=H_q2.eigenstates()
# [Eenergy_c,Estate_c]=H_ct(phi_e0)['H'].eigenstates()
# psi_00=tensor(Estate_q1[0],Estate_c[0],Estate_q2[0])
# psi_10=tensor(Estate_q1[1],Estate_c[0],Estate_q2[0])
# psi_01=tensor(Estate_q1[0],Estate_c[0],Estate_q2[1])
# psi_11=tensor(Estate_q1[1],Estate_c[0],Estate_q2[1])
# psi_20=tensor(Estate_q1[2],Estate_c[0],Estate_q2[0])
# psi_02=tensor(Estate_q1[0],Estate_c[0],Estate_q2[2])
# psi_00=E_state[0]
# psi_10=E_state[1]
# psi_01=E_state[2]
# psi_11=E_state[6]
# psi_20=E_state[4]
# psi_02=E_state[8]
psi_1=basis(N**4,1);psi_1.dims=[[8,8,8,8],[1]];psi_10=psi_1
psi_2=basis(N**4,2);psi_2.dims=[[8,8,8,8],[1]];psi_01=psi_2
psi_5=basis(N**4,4);psi_5.dims=[[8,8,8,8],[1]];psi_20=psi_5
psi_6=basis(N**4,6);psi_6.dims=[[8,8,8,8],[1]];psi_11=psi_6
psi_7=basis(N**4,8);psi_7.dims=[[8,8,8,8],[1]];psi_02=psi_7


# rho_00=psi_00*psi_00.dag()
rho_10=psi_10*psi_10.dag()
rho_01=psi_01*psi_01.dag()
rho_11=psi_11*psi_11.dag()
rho_20=psi_20*psi_20.dag()
rho_02=psi_02*psi_02.dag()
Gcz_list1=[]
Gcz_list2=[]
Giswap_list=[]
num_phi=61
Phi_er=np.linspace(-0.50,0.5,num_phi)
Phi_er=np.sort(np.append(Phi_er,phi_e0))
for i in range(len(Phi_er)):
    phi_e=Phi_er[i]
    H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e)['tensor_H']
    g1_i=psi_20.dag()*S_q.dag()*H_t*S_q*psi_11
    g2_i=psi_02.dag()*S_q.dag()*H_t*S_q*psi_11
    g_iswap=psi_10.dag()*S_q.dag()*H_t*S_q*psi_01
    # xi_i=E_energy[6]+E_energy[0]-E_energy[1]-E_energy[2]Ph
    Gcz_list1.append(g1_i.full()[0][0]/2/np.pi*1e3)
    Gcz_list2.append(g2_i.full()[0][0]/2/np.pi*1e3)
    Giswap_list.append(g_iswap.full()[0][0]/2/np.pi*1e3)
plt.figure()
plt.plot(Phi_er,Gcz_list1,label=r'$g_{eff,|20\rangle,|11\rangle}$')
plt.plot([phi_e0,phi_e0],[np.min(Gcz_list1),np.max(Gcz_list1)],'k--')
plt.xlabel(r'$\phi_e$')
plt.ylabel(r'$g_{eff}$ (MHz)')
plt.legend()
plt.figure()
plt.plot(Phi_er,Gcz_list2,label=r'$g_{eff,|02\rangle,|11\rangle}$')
plt.plot([phi_e0,phi_e0],[np.min(Gcz_list2),np.max(Gcz_list2)],'k--')
plt.xlabel(r'$\phi_e$')
plt.ylabel(r'$g_{eff}$ (MHz)')
plt.legend()
plt.figure()
plt.plot(Phi_er,Giswap_list,label=r'$g_{eff,|01\rangle,|10\rangle}$')
plt.plot([phi_e0,phi_e0],[np.min(Giswap_list),np.max(Giswap_list)],'k--')
plt.xlabel(r'$\phi_e$')
plt.ylabel(r'$g_{eff}$ (MHz)')
plt.legend()
index=[abs(i-5) for i in Gcz_list1].index(np.min(np.abs([i-5 for i in Gcz_list1])))
print(Phi_er[index])


#%% 测量能谱
num_phi=21
Phi_er=np.linspace(-0.0,0.5,num_phi)
# E_list=[]
def multy_En(k):
    # print(i)
    # phi_e=Phi_er[i]
    phi_e=Phi_er[k]
    H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e)['tensor_H']
    H_C = H_ct(phi_e)['H']
    # H_t=H_C0t+H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e)['tensor_H']
    # H_t=H_ct(phi_e)['Hiu']
    E_energy=H_t.eigenenergies()
    Ec_energy=H_C.eigenenergies()
    # [E_energy,E_state]=H_ct(phi_e)['H'].eigenstates()
    # E_list.append(E_energy)
    return(E_energy,Ec_energy)

E_list=Parallel(n_jobs=10,verbose=4)(delayed(multy_En)(k) for k in range(num_phi))


plt.figure()
for i in range(12):
    plt.plot(Phi_er,[(E_list[j][0][i+1]-E_list[j][0][0])/2/np.pi for j in range(len(Phi_er))])
plt.ylabel(r'Frequency (GHZ)')
plt.xlabel(r'$\phi_e$')


plt.figure()
for i in range(5):
    plt.plot(Phi_er,[(E_list[j][1][i+1]-E_list[j][1][0])/2/np.pi for j in range(len(Phi_er))])
plt.ylabel(r'Frequency (GHZ)')
plt.xlabel(r'$\phi_e$')

plt.figure()
plt.plot(Phi_er,[(E_list[j][0][6]+E_list[j][0][0]-E_list[j][0][1]-E_list[j][0][2])/2/np.pi*1e6 for j in range(len(Phi_er))])
plt.ylim([-20,20])
plt.ylabel(r'Frequency (KHZ)')
plt.xlabel(r'$\phi_e$')
# #%% 扫描ZZ耦合零点
# # EJ_list=np.linspace(12,22,41)*2*np.pi

# # def Find_psi11_index(psi_i,TensorState_list):
# #     P_list=[]
# #     for i in range(len(TensorState_list)):
# #         P_i=(psi_i*psi_i.dag()*TensorState_list[i]*TensorState_list[i].dag()).tr()
# #         P_list.append(P_i)
# #     index_i=P_list.index(np.max(P_list))
# #     # psi_index=TensorState_list[index_i]
# #     return(index_i)
# # num_phi=31
# Phi_er=np.linspace(0,0.4,num_phi)
# # def H_ct_EC(phi_ei,EJ_c1):
# #     print(EJ_c1)
# #     phi_e=phi_ei*2*np.pi
# #     EJ_c1p=EJ_c1+EJ_12*np.cos(phi_e)
# #     EJ_c2p=EJ_c2+EJ_12*np.cos(phi_e)
# #     # EJ_12p=EJ_12*np.cos(phi_e12)
# #     n_zpf_c1=((EJ_c1p)/(32*EC_c1))**0.25
# #     n_zpf_c2=((EJ_c2p)/(32*EC_c2))**0.25
# #     phi_zpf_c1=(2*EC_c1/(EJ_c1p))**0.25
# #     phi_zpf_c2=(2*EC_c2/(EJ_c2p))**0.25
# #     n_c1=tensor(-n_zpf_c1*1j*(a-a_d),qeye(N))
# #     n_c2=tensor(qeye(N),-n_zpf_c2*1j*(a-a_d))
# #     phi_c1=tensor(phi_zpf_c1*(a+a_d),qeye(N))
# #     phi_c2=tensor(qeye(N),phi_zpf_c2*(a+a_d))
# #     dphi_e12=0
    
# #     H_C=4*EC_c1*n_c1**2+4*EC_c2*n_c2**2+4*EC_12*n_c1*n_c2
# #     # H_phi12=EJ_12p*dphi_e12/EJ_c1p*(C_c1*4*EC_c1*n_c1/e+C_c1*2*EC_12/e*n_c2)-\
# #     #     EJ_12p*dphi_e12/EJ_c2p*(C_c2*4*EC_c2*n_c2/e+C_c2*2*EC_12/e*n_c1)
# #     H_J=-EJ_c1*phi_c1.cosm()-EJ_c2*phi_c2.cosm()-EJ_12*(np.cos(phi_e)*(phi_c2-phi_c1).cosm()+np.sin(phi_e)*(phi_c2-phi_c1).sinm())
# #     H=H_C+H_J
# #     H_tensor=tensor(qeye(N),H,qeye(N))
# #     return({'H':H, 'tensor_H':H_tensor})
# # def multy_job(k):
# #     EJ_c1=EJ_list[k]
# #     Xi_list=[]
# #     for i in range(num_phi):
# #         phi_e=Phi_er[i]
# #         H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct_EC(phi_e,EJ_c1)['tensor_H']
# #         [E_energy,E_state]=H_t.eigenstates()
# #         # E_energy=H_t.eigenenergies()
# #         [Eenergy_q1,Estate_q1]=H_q1.eigenstates()
# #         [Eenergy_q2,Estate_q2]=H_q2.eigenstates()
# #         [Eenergy_c,Estate_c]=H_ct_EC(phi_e,EJ_c1)['H'].eigenstates()
# #         psi_10=tensor(Estate_q1[1],Estate_c[0],Estate_q2[0])
# #         psi_01=tensor(Estate_q1[0],Estate_c[0],Estate_q2[1])
# #         psi_11=tensor(Estate_q1[1],Estate_c[0],Estate_q2[1])
# #         index_01=Find_psi11_index(psi_01,E_state[0:6])
# #         index_10=Find_psi11_index(psi_10,E_state[0:6])
# #         index_11=Find_psi11_index(psi_11,E_state[0:10])
# #         xi_i=E_energy[index_11]+E_energy[0]-E_energy[index_01]-E_energy[index_10]
# #         Xi_list.append(xi_i/2/np.pi)
# #     return(Xi_list)
# # data1 = Parallel(n_jobs=7, verbose=2)(delayed(multy_job)(k) for k in range(len(EJ_list)))
# # plt.figure()
# # X,Y=np.meshgrid(Phi_er,EJ_list/2/np.pi)
# # plt.pcolor(X,Y,np.real(data1)*1e6,cmap='jet')
# # plt.clim(-1,1)
# # plt.colorbar()
# # plt.xlabel(r'$\phi_e$')
# # plt.ylabel(r'$E_{J2}$')
# # plt.figure()
# # plt.plot(Phi_er,[i*1e6 for i in Xi_list],label='ZZ coupling')
# # plt.xlabel(r'$\phi_e$')
# # plt.ylabel(r'$\xi_{zz}$ kHz')
# # plt.plot([np.min(Phi_er),np.max(Phi_er)],[0,0],'k--')
# # plt.legend()
# e=1.6021766208e-19
# h=6.626070154e-34
# h_bar=h/2/np.pi
# C_c1=60e-15 #fF
# C_c2=60e-15 #fF
# C_12=10e-15 #fF
# C_t=np.sqrt(C_c1*C_c2+C_c1*C_12+C_c2*C_12)
# EC_c1=(C_c2+C_12)*e**2/(2*C_t**2)/h_bar/1e9
# EC_c2=(C_c1+C_12)*e**2/(2*C_t**2)/h_bar/1e9
# EC_12=(C_12)*e**2/(C_t**2)/h_bar/1e9
# EJ_c1=18*2*np.pi #7GHZ
# EJ_c2=26*2*np.pi #8.5GHZ
# EJ_12=(EJ_c1+EJ_c2)/4
# def H_ct_EC(phi_ei,EJ_12):
#     print(EJ_12)
#     phi_e=phi_ei*2*np.pi
#     EJ_c1p=EJ_c1+EJ_12*np.cos(phi_e)
#     EJ_c2p=EJ_c2+EJ_12*np.cos(phi_e)
#     # EJ_12p=EJ_12*np.cos(phi_e12)
#     n_zpf_c1=((EJ_c1p)/(32*EC_c1))**0.25
#     n_zpf_c2=((EJ_c2p)/(32*EC_c2))**0.25
#     phi_zpf_c1=(2*EC_c1/(EJ_c1p))**0.25
#     phi_zpf_c2=(2*EC_c2/(EJ_c2p))**0.25
#     n_c1=tensor(-n_zpf_c1*1j*(a-a_d),qeye(N))
#     n_c2=tensor(qeye(N),-n_zpf_c2*1j*(a-a_d))
#     phi_c1=tensor(phi_zpf_c1*(a+a_d),qeye(N))
#     phi_c2=tensor(qeye(N),phi_zpf_c2*(a+a_d))
#     dphi_e12=0
    
#     H_C=4*EC_c1*n_c1**2+4*EC_c2*n_c2**2+4*EC_12*n_c1*n_c2
#     # H_phi12=EJ_12p*dphi_e12/EJ_c1p*(C_c1*4*EC_c1*n_c1/e+C_c1*2*EC_12/e*n_c2)-\
#     #     EJ_12p*dphi_e12/EJ_c2p*(C_c2*4*EC_c2*n_c2/e+C_c2*2*EC_12/e*n_c1)
#     H_J=-EJ_c1*phi_c1.cosm()-EJ_c2*phi_c2.cosm()-EJ_12*(np.cos(phi_e)*(phi_c2-phi_c1).cosm()+np.sin(phi_e)*(phi_c2-phi_c1).sinm())
#     H=H_C+H_J
#     H_tensor=tensor(qeye(N),H,qeye(N))
#     return({'H':H, 'tensor_H':H_tensor})
# EJ_list=np.linspace(8,15,15)*2*np.pi
# def multy_job(k):
#     EJ_12=EJ_list[k]
#     Xi_list=[]
#     for i in range(num_phi):
#         phi_e=Phi_er[i]
#         H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct_EC(phi_e,EJ_12)['tensor_H']
#         [E_energy,E_state]=H_t.eigenstates()
#         # E_energy=H_t.eigenenergies()
#         [Eenergy_q1,Estate_q1]=H_q1.eigenstates()
#         [Eenergy_q2,Estate_q2]=H_q2.eigenstates()
#         [Eenergy_c,Estate_c]=H_ct_EC(phi_e,EJ_12)['H'].eigenstates()
#         psi_10=tensor(Estate_q1[1],Estate_c[0],Estate_q2[0])
#         psi_01=tensor(Estate_q1[0],Estate_c[0],Estate_q2[1])
#         psi_11=tensor(Estate_q1[1],Estate_c[0],Estate_q2[1])
#         index_01=Find_psi11_index(psi_01,E_state[0:6])
#         index_10=Find_psi11_index(psi_10,E_state[0:6])
#         index_11=Find_psi11_index(psi_11,E_state[0:10])
#         xi_i=E_energy[index_11]+E_energy[0]-E_energy[index_01]-E_energy[index_10]
#         Xi_list.append(xi_i/2/np.pi)
#     return(Xi_list)
# data1 = Parallel(n_jobs=7, verbose=2)(delayed(multy_job)(k) for k in range(len(EJ_list)))
# plt.figure()
# X,Y=np.meshgrid(Phi_er,EJ_list/2/np.pi)
# plt.pcolor(X,Y,np.real(data1)*1e6,cmap='jet')
# plt.clim(-0.5,0.5)
# plt.colorbar()
# plt.xlabel(r'$\phi_e$')
# plt.ylabel(r'$E_{J12}$')






