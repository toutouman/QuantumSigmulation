# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 15:19:19 2021

@author: 馒头你个史剃磅
"""



from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
from scipy import interpolate #插值优化

num=101
g=0.0125*2*np.pi
omega_1=4.85*2*np.pi
omega_2=4.885*2*np.pi
d_omega=omega_1-omega_2
t_list=np.linspace(0,200,num)
psi_0=tensor(basis(2),basis(2))
psi_1=tensor(basis(2,0),basis(2,1))
psi_2=tensor(basis(2,1),basis(2))
psi_3=tensor(basis(2,1),basis(2,1))

H_0=omega_1/2*tensor(sigmaz(),qeye(2))+omega_2/2*tensor(qeye(2),sigmaz())
H_i=g*tensor(sigmay(),sigmay())
H=H_0+H_i

# H_pm=tensor(create(2),destroy(2))
# H_mp=tensor(destroy(2),create(2))
# def pm_coffe(t,arg):
#     return(g*np.exp(1j*d_omega*t))
# def mp_coffe(t,arg):
#     return(g*np.exp(-1j*d_omega*t))
# H=[[H_pm,pm_coffe],[H_mp,mp_coffe]]
# result=mesolve(H,psi_1,t_list,[],[psi_0*psi_0.dag(),psi_1*psi_1.dag(),psi_2*psi_2.dag()])
# P_0=result.expect[0]
# P_1=result.expect[1]
# P_2=result.expect[2]

# plt.figure()
# # plt.plot(t_list,P_0,label=r'$P_0$')
# plt.plot(t_list,P_1,label=r'$P_1$')
# # plt.plot(t_list,P_2,label=r'$P_2$')
# plt.legend()

omega_range=np.linspace(0,0.03,num)

def multijob(i):
    omega_1=(4.87+omega_range[i])*2*np.pi
    H_0=omega_1/2*tensor(sigmaz(),qeye(2))+omega_2/2*tensor(qeye(2),sigmaz())
    H_i=g*tensor(sigmay(),sigmay())
    H=H_0+H_i
    result=mesolve(H,psi_1,t_list,[],[psi_0*psi_0.dag(),psi_1*psi_1.dag(),psi_2*psi_2.dag()])
    P_1=result.expect[1]
    return P_1
data = Parallel(n_jobs=3, verbose=2)(delayed(multijob)(i) for i in range(num))
P1=np.array(data)

Y,X=np.meshgrid(t_list,omega_range+4.87)
cm=plt.cm.get_cmap('jet')
plt.figure()
fig_x = plt.pcolor(X,Y,P1,cmap=cm)
cbar=plt.colorbar()
plt.show()

# beta_rf=1
phi_a=0
def f(phi,beta_rf):
    y=(phi-2*np.pi*phi_a)**2/(2*beta_rf)-np.cos(phi)
    return(y)
Phi=np.linspace(-5,5,101)
plt.figure()
# plt.plot(Phi,[f(Phi[i]*np.pi,0.8) for i in range(101)])
plt.plot(Phi,[f(Phi[i]*np.pi,1) for i in range(101)])
plt.show()
    
    
    
    