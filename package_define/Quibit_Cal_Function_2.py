# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:12:29 2022

@author: mantoutou
"""



from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from scipy import optimize


#电荷表象下的电荷数哈密顿
def Hn_charge(N):
    n_range=np.linspace(0,N,N+1)
    H_t=0
    for i in range(N):
        n=int(n_range[i])
        H_c=(n-int((N/2)))*basis(N,n)*basis(N,n).dag()
        H_t=H_t+H_c
    return(H_t)

#电荷表象下的单比特哈密顿
def Hqubit_charge(E_C,E_J,N,n_g):
    n_range=np.linspace(0,N,N+1)
    H_j=-E_J/2*(qdiags([1]*(N-1),1)+qdiags([1]*(N-1),-1))
    H_t=H_j
    for i in range(N):
        n=int(n_range[i])
        H_c=(4*E_C*(n-n_g-(N/2))**2)*basis(N,n)*basis(N,n).dag()
        H_t=H_t+H_c
    return(H_t)
#电荷表象下的单比特能级及非谐性
def Cal_Qubits_Freq_ChargeF(E_C,E_J,out_level=1,N=30,n_g=0):
    out_level=int(out_level)
    N=int(N)
    n_range=np.linspace(0,N,N+1)
    H_n=Hn_charge(N)
    H_t=Hqubit_charge(E_C,E_J,N,n_g)
    [E,eigenstate]=H_t.eigenstates()
    Omega_list=[(E[i+1]-E[i]) for i in range(out_level)]
    alpha=((E[2]+E[0]-2*E[1]))
    return(Omega_list,alpha)

#利用比特顶点频率及非谐性计算E_C,E_J
def Cal_Qubit_ECJ(omega,alpha,para_start=[0.25*2*np.pi,12*2*np.pi]):
    def min_error(para_list):
        E_C=para_list[0]
        E_J=para_list[1]
        [[omega_i],alpha_i]=Cal_Qubits_Freq_ChargeF(E_C,E_J)
        error=(abs(omega-omega_i)+abs(alpha-alpha_i))
        return(error)
    bounds=((0,0),(100*2*np.pi,200*2*np.pi))
    para_list=[i for i in para_start]
    opt_result=optimize.minimize(min_error,x0=(para_start[0],para_start[1]),method='Nelder-Mead',callback=para_list.append,
                                 options = {'maxiter': 300,'xatol':1e-4,'fatol':1e-4,'bounds':bounds},tol=1e-4)
    EC_opt=opt_result['x'][0]
    EJ_opt=opt_result['x'][1]
    return(EC_opt,EJ_opt)