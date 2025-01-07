# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 18:52:15 2020

@author: 馒头你个史剃磅
"""


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
starttime=int(time.time())
N=50
E_C=1
E_J=50*E_C
num=81
# n_g=0.5
N_g=np.linspace(0,1,num)
n_range=np.linspace(0,N,N+1)
H_j=-E_J/2*(qdiags([1]*(N-1),1)+qdiags([1]*(N-1),-1))
def H(H_j,n_g):
    H_t=H_j
    for i in range(N):
        n=int(n_range[i])
        H_c=(4*E_C*(n-n_g-(N/2))**2)*basis(N,n)*basis(N,n).dag()
        H_t=H_t+H_c
    return(H_t)

def multy_job(k):
    n_g=N_g[k]
    H_t=H(H_j,n_g)
    E_01=(H(H_j,0.5).eigenenergies()[1]-H(H_j,0.5).eigenenergies()[0])
    E_0=H(H_j,0.5).eigenenergies()[0]
    e_i=H_t.eigenenergies()
    e_0=((e_i[0])-E_0)/E_01
    e_1=((e_i[1])-E_0)/E_01
    e_2=((e_i[2])-E_0)/E_01
    e_3=((e_i[3])-E_0)/E_01
    return(e_0,e_1,e_2,e_3)
data = Parallel(n_jobs=3, verbose=2)(delayed(multy_job)(k) for k in range(num))

E_0=[i[0] for i in data]
E_1=[i[1] for i in data]
E_2=[i[2] for i in data]
E_3=[i[3] for i in data]
Omega_01=[i[1]-i[0] for i in data]
Omega_12=[i[2]-i[1] for i in data]


"""""""""""""""""E_J<<E_C微扰近似"""""""""""""""""

# E_0sim=[]
# E_1sim=[]
# for i in range(num):
#     if N_g[i]%1==0.5:
#         e_0=4*E_C*(0.5)**2-E_J/2+E_J**2/4/(4*E_C*(0.5)**2-4*E_C*(1.5)**2)
#         e_1=4*E_C*(0.5)**2+E_J/2+E_J**2/4/(4*E_C*(0.5)**2-4*E_C*(1.5)**2)
#         # e_2=4*E_C(1.5)**2-E_J/2+E_J**2/4/(4*E_C(0.5)**2-4*E_C(1.5)**2)
#     else:
#         e_0=4*E_C*N_g[i]**2+E_J**2/(8*E_C*(4*N_g[i]**2-1))
#         e_1=4*E_C*(1-N_g[i])**2+E_J**2/(8*E_C*(4*(1-N_g[i])**2-1))
#         e_2=4*E_C*(2-N_g[i])**2+E_J**2/(8*E_C*(4*(2-N_g[i])**2-1))
#         e_3=4*E_C*(3-N_g[i])**2+E_J**2/(8*E_C*(4*(3-N_g[i])**2-1))
#         e_m1=4*E_C*(-1-N_g[i])**2+E_J**2/(8*E_C*(4*(-1-N_g[i])**2-1))
#         e_m2=4*E_C*(-2-N_g[i])**2+E_J**2/(8*E_C*(4*(-2-N_g[i])**2-1))
#         E_lim=sorted([e_0,e_1,e_2,e_3,e_m1,e_m2])
#         e_0=E_lim[0]
#         e_1=E_lim[1]
#     E_0sim.append(e_0)
#     E_1sim.append(e_1)

"""""""""""""""E_J>>E_C微扰近似"""""""""""""""""
# E_0sim=[-E_J+np.sqrt(8*E_J*E_C)*(0+0.5)-E_C*(6*0**2+6*0+3)/12 for i in range(num)]
# E_1sim=[-E_J+np.sqrt(8*E_J*E_C)*(1+0.5)-E_C*(6*1**2+6*0+3)/12 for i in range(num)]
# E_2sim=[-E_J+np.sqrt(8*E_J*E_C)*(2+0.5)-E_C*(6*2**2+6*0+3)/12 for i in range(num)]



plt.figure()
plt.plot(N_g,E_0,label=r'$E_0$')
plt.plot(N_g,E_1,label=r'$E_1$')
plt.plot(N_g,E_2,label=r'$E_2$')
# plt.plot([0.5,0.5],[min(E_0),max(E_2)],'k--',label=r'Sweet spot')
# plt.plot(N_g,E_3,label=r'$E_3$')
# plt.plot(N_g,E_0sim,label=r'$E_0$ by Perturbation')
# plt.plot(N_g,E_1sim,label=r'$E_1$ by Perturbation')
# plt.plot(N_g,E_2sim,label=r'$E_2$ by Perturbation')
plt.ylabel(r'Energy level')
plt.xlabel(r'Offset charge $n_g$')
plt.legend()


plt.figure()
plt.plot(N_g,Omega_01,label=r'$\omega_{01}$')
plt.plot(N_g,Omega_12,label=r'$\omega_{12}$')
plt.legend()



endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')