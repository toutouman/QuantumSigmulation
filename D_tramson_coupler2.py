# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:30:59 2023

@author: mantoutou
"""
#%%
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import optimize
from joblib import Parallel, delayed
from QubitHamiltonian import QubitDTCouplerQubit
#%%

starttime=int(time.time())

e=1.6021766208e-19
h=6.626070154e-34
h_bar=h/2/np.pi
C_q1 = 85e-15 #fF
C_q2 = 85e-15 #fF
C_c1=70e-15 #fF
C_c2=70e-15 #fF
C_c=30e-15 #fF

C_qc1 = 7.9e-15 
C_qc2 = 7.9e-15
C_12 = 0.23e-15

Ej_q1=23*2*np.pi
Ej_q2=26*2*np.pi
Ej_c1=41*2*np.pi 
Ej_c2=41*2*np.pi 
Ej_12=41*2*np.pi*0.2347

QCQ = QubitDTCouplerQubit(C_q1 = C_q1, C_q2 = C_q2, 
                          C_c1 = C_c1, C_c2 = C_c2, C_c = C_c,
                          C_qc1 = C_qc1, C_qc2 = C_qc2, C_12 = C_12,
                          Ej_q1 = Ej_q1, Ej_q2 = Ej_q2, 
                          Ej_c1 = Ej_c1, Ej_c2 = Ej_c2, Ej_c12 = Ej_12)
#%% 测量能谱
num_phi=21
Phi_er=np.linspace(-0.0,0.5,num_phi)
# E_list=[]
def multy_En(k):
    # print(i)
    # phi_e=Phi_er[i]
    phi_e=Phi_er[k]*2*np.pi
    Eigenenergies_sub_list, Eigenenergies = QCQ.H_eigenenergies(phi_ex_list = [0,phi_e,0])
    
    return(Eigenenergies_sub_list, Eigenenergies)

E_list=Parallel(n_jobs=10,verbose=4)(delayed(multy_En)(k) for k in range(num_phi))


plt.figure()
for i in range(12):
    plt.plot(Phi_er,[(E_list[j][1][i+1]-E_list[j][1][0])/2/np.pi for j in range(len(Phi_er))])
plt.ylabel(r'Frequency (GHZ)')
plt.xlabel(r'$\phi_e$')
#%%
num_phi=21
Phi_er=np.linspace(-0.0,0.5,num_phi)
# E_list=[]
def multy_En(k):
    # print(i)
    # phi_e=Phi_er[i]
    phi_e=Phi_er[k]*2*np.pi
    eigens_dict = QCQ.QCQ_eigens(phi_ex_list = [0,phi_e,0])
    
    return(eigens_dict)
E_list=Parallel(n_jobs=11,verbose=4)(delayed(multy_En)(k) for k in range(num_phi))

E_energies_lists = [datas['E_energies'] for datas in E_list] 
psi1_index_lists = [datas['psi1_index_list'] for datas in E_list]
Energy_psi1_lists = [[E[index] for index in psi1_index_lists[i]] for i,E in enumerate(E_energies_lists)]

psi101_index_lists = [datas['psi_101_index'] for datas in E_list]
Energy_psi101_lists = [E[psi101_index_lists[i]] for i,E in enumerate(E_energies_lists)]

plt.figure()
for i in range(12):
    plt.plot(Phi_er,[(E_energies_lists[j][i+1]-E_energies_lists[j][0])/2/np.pi for j in range(len(Phi_er))])
plt.ylabel(r'Frequency (GHZ)')
plt.xlabel(r'$\phi_e$')



ZZ_Strength = [(Energy_psi1_lists[j][0]+Energy_psi1_lists[j][-1]-
                E_energies_lists[j][0]-Energy_psi101_lists[j])/2/np.pi*1e6 for j in range(len(Phi_er))]
plt.figure()
plt.plot(Phi_er,ZZ_Strength)
#%%
Eenergy_sub_list,Estate_sub_list,E_energies,E_states =QCQ.H_eigenstates(phi_ex_list = [0,0.5*2*np.pi,0])
Eenergy_sub_q1 = Eenergy_sub_list[0]
Estate_sub_q1 = Estate_sub_list[0]
psi_1_sub_list = [Es[1] for Es in Estate_sub_list]
psi_2_sub_list = [Es[2] for Es in Estate_sub_list]
ptrace_index = [[0], [1,2], [3]]
P_lists = [[(psi_1*psi_1.dag()*E_s.ptrace(ptrace_index[i])).tr() for E_s in E_states[:20]]
           for i,psi_1 in enumerate(psi_1_sub_list)]
psi1_index_list = [np.argmax(p_l) for p_l in P_lists]
