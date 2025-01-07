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


"""电荷表象下的电荷数哈密顿"""
def Hn_charge(N):
    n_range=np.linspace(0,N,N+1)
    H_t=0
    for i in range(N):
        n=int(n_range[i])
        H_c=(n-int((N/2)))*basis(N,n)*basis(N,n).dag()
        H_t=H_t+H_c
    return(H_t)

"""电荷表象下的单比特哈密顿"""
def Hqubit_charge(E_C,E_J,N,n_g):
    n_range=np.linspace(0,N,N+1)
    H_j=-E_J/2*(qdiags([1]*(N-1),1)+qdiags([1]*(N-1),-1))
    H_t=H_j
    for i in range(N):
        n=int(n_range[i])
        H_c=(4*E_C*(n-n_g-(N/2))**2)*basis(N,n)*basis(N,n).dag()
        H_t=H_t+H_c
    return(H_t)

"""电荷表象下的单比特能级及非谐性"""
def Cal_Qubits_Freq_Charge(E_C,E_J,out_level=1,N=30,n_g=0):
    out_level=int(out_level)
    N=int(N)
    # print(N)
    n_range=np.linspace(0,N,N+1)
    H_n=Hn_charge(N)
    H_t=Hqubit_charge(E_C,E_J,N,n_g)
    [E,eigenstate]=H_t.eigenstates()
    Omega_list=[(E[i+1]-E[i]) for i in range(out_level)]
    alpha=((E[2]+E[0]-2*E[1]))
    return(Omega_list,alpha)

"""计算浮地哈密顿量"""

"""计算接地比特浮地coupler耦合Q-C-Q哈密顿量"""
def QCQ_GFG_H(C_1,C_4,C_c,C_02,C_03,
              C_13,C_12,C_24,C_34,
              N):
    e=1.60217e-19
    h=6.62607e-34
    h_bar=h/2/np.pi
    C_SP = C_03+C_02+C_13+C_12+C_34+C_24
    C_SM = C_03-C_02+C_13-C_12+C_34-C_24
    C_M = np.array([
        [C_1+C_12+C_13, -0.5*(C_12+C_13), -0.5*(C_13-C_12), 0],
        [-0.5*(C_12+C_13), C_SP/4, C_SM/4, -0.5*(C_24+C_34)],
        [-0.5*(C_13-C_12), C_SM/4, C_SP/4+C_c, -0.5*(C_34-C_24)],
        [0, -0.5*(C_34+C_24), -0.5*(C_34-C_24), C_4+C_34+C_24]
        ])
    C_inv = np.linalg.inv(C_M)
    
    Ec_1 = e**2/(2*C_inv[0,0])/h_bar/1e9    # GHz
    Ec_2 = e**2/(2*C_inv[3,3])/h_bar/1e9    # GHz
    Ec_c = e**2/(2*C_inv[2,2])/h_bar/1e9    # GHz
    Ec_12 = 2*e**2/(2*C_inv[0,3])/h_bar/1e9 # GHZ
    Ec_1c = 2*e**2/(2*C_inv[0,2])/h_bar/1e9 # GHZ
    Ec_2c = 2*e**2/(2*C_inv[3,2])/h_bar/1e9 # GHZ
    

"""利用比特顶点频率及非谐性计算E_C,E_J 单位GHz*2pi"""
def Cal_Qubit_ECJ(omega,alpha,para_start=[0.25*2*np.pi,12*2*np.pi]):
    def min_error(para_list):
        E_C=para_list[0]
        E_J=para_list[1]
        [[omega_i],alpha_i]=Cal_Qubits_Freq_Charge(E_C,E_J,n_g=0.25)
        error=(abs(omega-omega_i)+abs(alpha-alpha_i))
        return(error)
    bounds=((0,0),(100*2*np.pi,400*2*np.pi))
    para_list=[i for i in para_start]
    opt_result=optimize.minimize(min_error,x0=(para_start[0],para_start[1]),method='Nelder-Mead',callback=para_list.append,
                                 options = {'maxiter': 300,'xatol':1e-4,'fatol':1e-4},tol=1e-4)
    EC_opt=opt_result['x'][0]
    EJ_opt=opt_result['x'][1]
    return(EC_opt,EJ_opt)
#根据频率-zpa计算zpa2f01
# def zpa2f01(f01_list,zpa_list):
#     def zpa2f01_func(zpa,a,b,c):
#         return(a*zpa**2+b*zpa+c)
#     fit_func=optimize.curve_fit(zpa2f01, Time2_r, M2_i,[8,1],bounds=param_bounds,maxfev = 40000)

"""给定比特顶点频率、磁通，计算比特频率和斜率"""
def omega01_flux(phi_ex,   #0-0.5*np.pi
                 omega_max,
                 alpha):
    e=1.6021766208e-19
    h=6.626070154e-34
    [E_cm,E_jm]=[i/2/np.pi*1e9 for i in Cal_Qubit_ECJ(omega_max/1e9*2*np.pi,alpha/1e9*2*np.pi)]
    E_j0 = E_jm*np.cos(phi_ex)
    [omega01_0], alpha_0 = Cal_Qubits_Freq_Charge(E_cm,E_j0,n_g = 0.25)
    Phi_0=h/2/e
    #k单位： Hz/Phi_0
    k=-0.5*8*E_cm*E_j0*np.pi/Phi_0*np.sin(phi_ex)/\
        np.sqrt(8*E_cm*E_j0*np.cos(phi_ex))*Phi_0
    return(omega01_0,alpha_0,k)

"""计算顶点磁通和斜率"""
def flux_k(p_list,alpha,omega_0):
    a=p_list[0]
    b=p_list[1]
    c=p_list[2]
    amp_max=-b/2/a
    omega_max=a*amp_max**2+b*amp_max+c
    delta=omega_max-omega_0
    amp_0=np.sqrt((omega_0-c)/a+(b/2/a)**2)-b/2/a
    k_bias=2*a*amp_0+b
    e=1.6021766208e-19
    h=6.626070154e-34
    [E_c,E_j]=[i/2/np.pi*1e9 for i in Cal_Qubit_ECJ(omega_max/1e9*2*np.pi,alpha/1e9*2*np.pi)]
    print(omega_max)
    # E_c=e**2/2/C/h
    # E_j=(omega_max+E_c)**2/8/E_c
    Phi_0=h/2/e
    Phi_ex=np.arccos((omega_0+E_c)**2/(8*E_c*E_j))*Phi_0/np.pi
    # Phi_ex=Phi_0*(delta+np.sqrt(8*E_c*E_j))**2/(8*E_c*E_j)/np.pi
    k=-0.5*8*E_c*E_j*np.pi/Phi_0*np.sin(np.pi*Phi_ex/Phi_0)/\
        np.sqrt(8*E_c*E_j*np.cos(np.pi*Phi_ex/Phi_0))*Phi_0
    return (Phi_ex/Phi_0,k)

"""利用顶点频率计算当前磁通斜率 (单位：Hz)"""
def flux_k_max(omega_max,alpha,omega_0):
    e=1.6021766208e-19
    h=6.626070154e-34
    [E_c,E_j]=[i/2/np.pi*1e9 for i in Cal_Qubit_ECJ(omega_max/1e9*2*np.pi,alpha/1e9*2*np.pi)]
    print(omega_max)
    # E_c=e**2/2/C/h
    # E_j=(omega_max+E_c)**2/8/E_c
    Phi_0=h/2/e
    Phi_ex=np.arccos((omega_0+E_c)**2/(8*E_c*E_j))*Phi_0/np.pi
    # Phi_ex=Phi_0*(delta+np.sqrt(8*E_c*E_j))**2/(8*E_c*E_j)/np.pi
    #k单位： Hz/Phi_0
    k=-0.5*8*E_c*E_j*np.pi/Phi_0*np.sin(np.pi*Phi_ex/Phi_0)/\
        np.sqrt(8*E_c*E_j*np.cos(np.pi*Phi_ex/Phi_0))*Phi_0
    return (Phi_ex/Phi_0,k)

"""利用顶点频率计算当前磁通斜率 (单位：Hz)"""
def flux_k_max(omega_max,alpha,omega_0):
    e=1.6021766208e-19
    h=6.626070154e-34
    [E_cm,E_jm]=[i/2/np.pi*1e9 for i in Cal_Qubit_ECJ(omega_max/1e9*2*np.pi,alpha/1e9*2*np.pi)]
    [E_c,E_j]=[i/2/np.pi*1e9 for i in Cal_Qubit_ECJ(omega_0/1e9*2*np.pi,alpha/1e9*2*np.pi)]
    # print((E_cm-E_c)/1e6)
    # E_c=e**2/2/C/h
    # E_j=(omega_max+E_c)**2/8/E_c
    Phi_0=h/2/e
    # Phi_ex=np.arccos((omega_0+E_c)**2/(8*E_c*E_j))*Phi_0/np.pi
    Phi_ex=np.arccos(E_j/E_jm)*Phi_0/np.pi
    #k单位： Hz/Phi_0
    k=-0.5*8*E_c*E_jm*np.pi/Phi_0*np.sin(np.pi*Phi_ex/Phi_0)/\
        np.sqrt(8*E_c*E_jm*np.cos(np.pi*Phi_ex/Phi_0))*Phi_0
    return (Phi_ex/Phi_0,k)

"""利用顶点频率近似计算当前斜率，速度更快 (单位：Hz)"""
def flux_k_max_sim(omega_max,alpha,omega_0):
    e=1.6021766208e-19
    h=6.626070154e-34
    E_c=-alpha
    E_jm=(omega_max+E_c)**2/8/E_c
    Phi_0=h/2/e
    Phi_ex=np.arccos((omega_0+E_c)**2/(8*E_c*E_jm))*Phi_0/np.pi
    #k单位： Hz/Phi_0
    k=-0.5*8*E_c*E_jm*np.pi/Phi_0*np.sin(np.pi*Phi_ex/Phi_0)/\
        np.sqrt(8*E_c*E_jm*np.cos(np.pi*Phi_ex/Phi_0))*Phi_0
    return (Phi_ex/Phi_0,k)

"""利用chi Delta eta 计算比特和读取腔耦合强度"""
def Cal_g_rq(chi,    # 比特0态腔频减去1态腔频/2，正数，HZ
             Delta,  # 比特频率减去读取腔频，负数，HZ
             eta, # 比特失谐，负数，HZ
             ):
    g=(-chi*Delta*(1+Delta/eta))**0.5
    return(g)

"""利用 g Delta omega_r omega_q omega_f kappa_f kappa_R 计算T1 (ms)上限"""
def Cal_purcell_T1(
        g,         #比特和读取腔耦合强度  Hz
        omega_r,   #读取腔频率  Hz 
        omega_q,   #比特频率   Hz
        omega_f,   #filter中心频率    Hz
        kappa_f,   #filter S21 半高全宽， 即顶点下降3dB处(1.414)的宽度   Hz
        kappa_r,   #读取腔kappa   Hz
        ):
    Delta=omega_r-omega_q
    T1=(Delta/g)**2*(omega_r/omega_q)*(omega_f*Delta/(omega_r*kappa_f/2))**2/kappa_r
    return(T1/1e-3)




