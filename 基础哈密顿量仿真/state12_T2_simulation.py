# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 21:53:05 2022

@author: mantoutou
"""


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
starttime=int(time.time())

alpha=0.25e2*2*np.pi
E_0=0
E_1=4e2*2*np.pi
E_2=2*E_1-alpha
N=3
times = np.linspace(0.0, 45, 51)
d_omega=0.5*2*np.pi

Gamma=1/64.12
Gamma_02=0.1

psi_0=basis(N,0)
psi_1=basis(N,1)
psi_2=basis(N,2)
rho_0=tensor(psi_0*psi_0.dag(),basis(2)*basis(2).dag())
rho_1=tensor(psi_1*psi_1.dag(),basis(2)*basis(2).dag())
rho_2=tensor(psi_2*psi_2.dag(),basis(2)*basis(2).dag())
# sigmaz_12=rho_1-rho_2
# sigmay_tls=tensor(qeye(N),sigmay())
H_c=tensor(1*psi_2*psi_2.dag()-1*psi_1*psi_1.dag(),sigmax())

a_d=tensor(create(N),qeye(2))
a=tensor(destroy(N),qeye(2))
a_2=tensor(destroy(N),qeye(2))

Ysq_operator=tensor((psi_0*psi_0.dag()-1j*psi_1*psi_2.dag()+1j*psi_2*psi_1.dag()).sqrtm(),qeye(2))
# Y_operator=tensor(psi_0*psi_0.dag()+1*psi_1*psi_2.dag()+1*psi_2*psi_1.dag(),qeye(2))
Y_operator=tensor(qeye(3),qeye(2))

H=0.5*d_omega*(tensor(1*psi_2*psi_2.dag()-1*psi_1*psi_1.dag(),qeye(2)))+Gamma_02*H_c
psi_int=tensor((1*psi_1-1*psi_2).unit(),basis(2))
rho_int=psi_int*psi_int.dag()
result = mesolve(H, rho_int, times, [],[])

P_2=[]
P_1=[]
P_0=[]
for i in range(51):
    rho_i0=Y_operator*result.states[i]*Y_operator.dag()
    # print(ptrace(result.states[25],[0,1]))
    # print(ptrace(rho_i0,[0,1]))
    t_i=times[i]
    if i==0:
        rho_i=rho_i0
    else:
        result_i=mesolve(H, rho_i0, np.linspace(0,t_i,51), [],[])
        rho_i=result_i.states[-1]
    p_2=psi_2.dag()*(Ysq_operator*rho_i*Ysq_operator.dag()).ptrace(0)*psi_2
    p_1=psi_1.dag()*(Ysq_operator*rho_i*Ysq_operator.dag()).ptrace(0)*psi_1
    p_0=psi_0.dag()*(Ysq_operator*rho_i*Ysq_operator.dag()).ptrace(0)*psi_0
    P_2.append(p_2[0][0])
    P_1.append(p_1[0][0])
    P_0.append(p_0[0][0])

plt.figure()
plt.plot(times, P_0,label='P_0')
plt.plot(times, P_1,label='P_1')
plt.plot(times, P_2,label='P_2')
plt.legend()
