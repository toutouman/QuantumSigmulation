# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:55:00 2023

@author: mantoutou
"""

#%% 导入包
import sys
sys.path.append(r'C:\Users\mantoutou\OneDrive\文档\程序\科大\约瑟夫森结哈密顿')
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import optimize
from joblib import Parallel, delayed
from package_define.Quibit_Cal_Function import flux_k_max_sim, flux_k_max
from package_define.QubitHamiltonian import GQubitGCouplerGQubit
def forward_diff(f,x,):  
    dx = abs(0.0001)*x
    """定义求导函数"""
    return((f(x+0.5*dx)-f(x-0.5*dx))/dx)
#%% 比特基本信息设置

starttime=int(time.time())

e=1.6021766208e-19
h=6.626070154e-34
h_bar=h/2/np.pi
C_q1=78e-15
C_q2=78e-15
C_c=125e-15 #fF

C_qc1 = 10.2e-15 
C_qc2 = 10.2e-15
C_12 = 0.34e-15

#ZCZ3.1 设计值
Rj_q1=8100
Rj_q2=8400
Rj_c = 1350

#P18V4 Al-Ta实际值
# Rj_q1=9400
# Rj_q2=9800
# Rj_c = 1400
QCQ = GQubitGCouplerGQubit(
            C_q1 = C_q1, C_q2 = C_q2, C_c = C_c,
            C_qc1 = C_qc1, C_qc2 = C_qc2, C_12 = C_12,
            Rj_q1 = Rj_q1, Rj_q2 = Rj_q2, Rj_c = Rj_c,
            N_level = 8,)
print(QCQ.Ec_Matrix)
#%% 计算比特频率等信息
phi_ex_q1 = 0
phi_ex_c = 0.0*np.pi
phi_ex_q2 = 0.18*np.pi
phi_ex_list = [phi_ex_q1,phi_ex_c,phi_ex_q2]
QCQsys_eigens = QCQ.QCQ_eigens(phi_ex_list)
QCQcouple_H = QCQ.Coupling_H(phi_ex_list)
H_sys = QCQcouple_H['H_total']
Eenergy_sub_list = QCQsys_eigens['Eenergy_sub_list']
Estate_sub_list = QCQsys_eigens['Estate_sub_list']
psi1_index_list = QCQsys_eigens['psi1_index_list']
psi1_tensor_list = QCQsys_eigens['psi1_tensor_list']
psi1_list = QCQsys_eigens['psi1_list']
E_energies = QCQsys_eigens['E_energies']
E_states = QCQsys_eigens['E_states']

index_q1 = 0
index_q2 = 2
index_c = 1
fre_q1_sub = [(Eenergy_sub_list[index_q1][i+1] - Eenergy_sub_list[index_q1][0])/2/np.pi
                 for i in range(len(Eenergy_sub_list[index_q1])-1)]
fre_c_sub = [(Eenergy_sub_list[index_c][i+1] - Eenergy_sub_list[index_c][0])/2/np.pi
                 for i in range(len(Eenergy_sub_list[index_c])-1)]
fre_q2_sub = [(Eenergy_sub_list[index_q2][i+1] - Eenergy_sub_list[index_q2][0])/2/np.pi
                 for i in range(len(Eenergy_sub_list[index_q2])-1)]
omega01_q1 = E_energies[psi1_index_list[index_q1]]- E_energies[0]
omega01_c = E_energies[psi1_index_list[index_c]]- E_energies[0]
omega01_q2 = E_energies[psi1_index_list[index_q2]]- E_energies[0]
omega_sys_list = [E_energies[i] - E_energies[0] for i in range(len(E_energies)-1)]

print(' f_q1_sub =', fre_q1_sub[0],'GHz\n f_q2_sub =',
      fre_q2_sub[0],'GHz\n f_c_sub =',fre_c_sub[0])
print(' f_q1 =', omega01_q1/2/np.pi,'GHz\n f_q2 =',
      omega01_q2/2/np.pi,'GHz\n f_c =',omega01_c/2/np.pi)
psi0_q1sub = Estate_sub_list[index_q1][0]
psi1_q1sub = Estate_sub_list[index_q1][1]
psi0_q2sub = Estate_sub_list[index_q2][0]
psi1_q2sub = Estate_sub_list[index_q2][1]
psi0_csub = Estate_sub_list[index_c][0]
psi1_csub = Estate_sub_list[index_c][1]

psi_000 = E_states[0]
psi_100 = E_states[psi1_index_list[index_q1]]
psi_010 = E_states[psi1_index_list[index_c]]
psi_001 = E_states[psi1_index_list[index_q2]]
g_qq = (psi1_tensor_list[index_q1]*psi1_tensor_list[index_q2].dag()*H_sys).tr()
g_qc1 = (psi1_tensor_list[index_q1]*psi1_tensor_list[index_c].dag()*H_sys).tr()
g_qc2 = (psi1_tensor_list[index_q2]*psi1_tensor_list[index_c].dag()*H_sys).tr()

print (' g_qq =', g_qq/2/np.pi,'GHz\n g_qc1=',
       g_qc1/2/np.pi,'GHz\n g_qc2=',g_qc2/2/np.pi,'GHz')
# print(g_qq**2/(omega01_q2-omega01_q1)/2/np.pi)
# print(g_qq**2/(omega01_q2-omega01_q1)**2)
psi0_q1 = (psi_000+psi_100).unit()
psi0_q2 = (psi_000+psi_001).unit()
#%% 计算等效耦合强度
starttime=int(time.time())
phi_ex_clist = np.linspace(np.pi*0.0,np.pi*0.40,81)
phi_ex_q1 = 0
phi_ex_q2 = 0

QCQ_geff = QCQ.Cal_geff(phi_ex_clist,
                        phi_ex_q1 = phi_ex_q1,
                        phi_ex_q2 = phi_ex_q2)
QCQ_eigens_list = QCQ_geff['QCQ_eigens_list']
# g_xy_list_leakage = QCQ_geff['g_xy_list_leakage']
g_xy_list = QCQ_geff['g_xy_list']
omega01_lists = QCQ_geff['omega01_lists']
g_zz_list = QCQ_geff['g_zz_list']

off_xy_index = QCQ_geff['off_xy_index']
off_zz_index = QCQ_geff['off_zz_index']
leakage_cToq1_list = QCQ_geff['leakage_cToq1_list']
leakage_cToq2_list = QCQ_geff['leakage_cToq2_list']
omega01_off_list = omega01_lists[off_xy_index]  #关断点的01频率列表
omega01_q1_list,omega01_c_list,omega01_q2_list = list(map(list, zip(*omega01_lists))) #行列互换得到各自的频率列表
E_energies_list = [eigen['E_energies'] for eigen in QCQ_eigens_list]

"""耦合强度随coupler频率变化"""
plt.figure()
plt.plot([o_list[1]/2/np.pi for o_list in omega01_lists],
         [abs(g)/1e-3/2/np.pi for g in g_xy_list],
          label = 'XY coupling strength')
plt.plot([omega01_lists[off_xy_index][1]/2/np.pi, omega01_lists[off_xy_index][1]/2/np.pi],
         [np.min(g_xy_list)/1e-3/2/np.pi, np.max(g_xy_list)/1e-3/2/np.pi], 'r--',
         label = 'XY-off point')
plt.plot([o_list[1]/2/np.pi for o_list in omega01_lists],
          [abs(g)/1e-3/2/np.pi for g in g_zz_list], 
          label = 'ZZ coupling strength')
plt.plot([omega01_lists[off_zz_index][1]/2/np.pi, omega01_lists[off_zz_index][1]/2/np.pi],
          [np.min(np.abs(g_zz_list))/1e-3/2/np.pi, np.max(np.abs(g_zz_list))/1e-3/2/np.pi], 'b--',
          label = 'ZZ-off point')
plt.xlabel('Coupler frequencies')
plt.ylabel('Effective qubit-qubit coupling strength (MHz)')
plt.yscale('log')
plt.legend()

"""比特频率随coupler频率变化"""
plt.figure()
plt.plot([o_list[1]/2/np.pi for o_list in omega01_lists],
         [o_list[0]/2/np.pi for o_list in omega01_lists],
         label = 'Qubit1 F01')
plt.plot([o_list[1]/2/np.pi for o_list in omega01_lists],
         [o_list[2]/2/np.pi for o_list in omega01_lists],
         label = 'Qubit2 F01')
plt.xlabel('Coupler frequencies')
plt.ylabel('Qubits frequencies (GHz)')
# plt.yscale('log')
plt.legend()
endtime=int(time.time())
print('Run-cell time is', endtime-starttime, 's\n')

#%% 计算比特频率等参数随coupler偏置变化
starttime=int(time.time())

phi_ex_clist = np.linspace(np.pi*0.1,np.pi*0.41,83)
phi_ex_q1 = 0
phi_ex_q2 = 0.2*np.pi

index_q1 = 0
index_q2 = 2
index_c = 1

QCQ_geff = QCQ.Cal_geff(phi_ex_clist,
                        phi_ex_q1 = phi_ex_q1,
                        phi_ex_q2 = phi_ex_q2)
QCQ_eigens_list = QCQ_geff['QCQ_eigens_list']
g_xy_list = QCQ_geff['g_xy_list']
off_xy_index = QCQ_geff['off_xy_index']
omega01_lists = QCQ_geff['omega01_lists']
g_zz_list = QCQ_geff['g_zz_list']
leakage_cToq1_list = QCQ_geff['leakage_cToq1_list']
leakage_cToq2_list = QCQ_geff['leakage_cToq2_list']
omega01_off_list = omega01_lists[off_xy_index]  #关断点的01频率列表
omega01_q1_list,omega01_c_list,omega01_q2_list = list(map(list, zip(*omega01_lists))) #行列互换得到各自的频率列表

omega_q1sub_lists = [[omega - e['Eenergy_sub_list'][index_q1][0]
                     for omega in e['Eenergy_sub_list'][index_q1][1:]]
                    for e in QCQ_eigens_list]                                #无耦合的Q1频率列表
omega_csub_lists = [[omega - e['Eenergy_sub_list'][index_c][0]
                     for omega in e['Eenergy_sub_list'][index_c][1:]]
                    for e in QCQ_eigens_list]                                #无耦合的coupler频率列表
omega_q2sub_lists = [[omega - e['Eenergy_sub_list'][index_q2][0]
                     for omega in e['Eenergy_sub_list'][index_q2][1:]]
                    for e in QCQ_eigens_list]                                #无耦合的Q2频率列表

delta01_q1_list = [ o-omega01_off_list[index_q1] for o in omega01_q1_list]  #Q1频率与关断点的频率差
delta01_q2_list = [ o-omega01_off_list[index_q2] for o in omega01_q2_list]  #Q2频率与关断点的频率差
delta_qc1_list = [omega_c_sub[0]-omega_q1_sub[0] for omega_c_sub,omega_q1_sub 
                  in zip(omega_csub_lists,omega_q1sub_lists)]                 #比特1和coupler频率差
# delta_qc1_list = [omega_c-omega_q1 for omega_c,omega_q1
#                   in zip(omega01_c_list,omega01_q1_list)]                 #比特1和coupler频率差
delta_qc2_list = [omega_c_sub[0]-omega_q2_sub[0] for omega_c_sub,omega_q2_sub 
                  in zip(omega_csub_lists,omega_q2sub_lists)]                 #比特1和coupler频率差

H_sys_lists = [QCQ.Coupling_H([phi_ex_q1,phi_c,phi_ex_q2])['H_total']
               for phi_c in phi_ex_clist]
psi1_tensor_lists = [eigen['psi1_tensor_list'] for eigen in QCQ_eigens_list]  #无耦合下的1态列表
psi1_lists = [eigen['psi1_list'] for eigen in QCQ_eigens_list]          #耦合系统下的1态列表
g_qq_list = [abs((psi_list[index_q1]*psi_list[index_q2].dag()*H).tr())
             for psi_list,H in zip(psi1_tensor_lists, H_sys_lists)]          #Q1Q2直接耦合强度
g_qc1_list = [abs((psi_list[index_q1]*psi_list[index_c].dag()*H).tr())
              for psi_list,H in zip(psi1_tensor_lists, H_sys_lists)]          #Q1-C直接耦合强度
g_qc2_list = [abs((psi_list[index_q2]*psi_list[index_c].dag()*H).tr())
              for psi_list,H in zip(psi1_tensor_lists, H_sys_lists)]          #Q2-C直接耦合强度

# 比特在关断点的频率和无耦合时的01频率差
Q1delta01_peak_off = omega01_off_list[index_q1] - omega_q1sub_lists[off_xy_index][0]
Q2delta01_peak_off = omega01_off_list[index_q2] - omega_q2sub_lists[off_xy_index][0]
"""多项式拟合XY耦合强度随coupler频率变化"""
params_g_xy = np.polyfit(omega01_c_list,g_xy_list,deg = 31)
XYgeff_fit = lambda o: np.poly1d(params_g_xy)(o)

"""多项式拟合比特频率相对关断点的差值随coupler频率变化"""
params_delta_q1 = np.polyfit(omega01_c_list,np.log(-np.array(delta01_q1_list)-Q1delta01_peak_off),deg = 31)
Q1delta_fit = lambda o: -np.exp(np.poly1d(params_delta_q1)(o)) - Q1delta01_peak_off
params_delta_q2 = np.polyfit(omega01_c_list,np.log(-np.array(delta01_q2_list)-Q2delta01_peak_off),deg = 31)
Q2delta_fit = lambda o: -np.exp(np.poly1d(params_delta_q2)(o)) - Q2delta01_peak_off

"""多项式拟合比特缀饰态|100>中含有coupler裸态|1>的概率随coupler频率变化"""
params_leakage = np.polyfit(omega01_c_list,np.log(np.array(leakage_cToq1_list)),deg = 21)
Q1leakage_fit = lambda o: np.exp(np.poly1d(params_leakage)(o))

plt.figure()
plt.plot([omega/2/np.pi for omega in omega01_c_list],
          [(Q1delta_fit(omega))/2/np.pi/1e-3 for omega in omega01_c_list])
plt.plot([omega/2/np.pi for omega in omega01_c_list],
          [(Q2delta_fit(omega))/2/np.pi/1e-3 for omega in omega01_c_list])
plt.plot([omega/2/np.pi for omega in omega01_c_list],
          [g/2/np.pi/1e-3 for g in g_xy_list])

# plt.plot([omega/2/np.pi for omega in omega01_c_list],
#           [-(g**2/delta+g**2/(delta+2*omega_q1sub_lists[0][0]))/2/np.pi/1e-3+33 for g,delta in zip(g_qc1_list,delta_qc1_list)])
# plt.plot([o_list[1]/2/np.pi for o_list in omega01_lists],
#           [g/1e-3/2/np.pi for g in g_xy_list],) 

omega01_c_range = np.linspace(np.min(omega01_c_list),np.max(omega01_c_list),1001)   #插值更多的频率
"""求解比特频率相对coupler频率变化的导数"""
delta01_q1_range = Q1delta_fit(omega01_c_range)/2/np.pi/1e-3
delta01_q2_range = Q2delta_fit(omega01_c_range)/2/np.pi/1e-3
geff_xy_range = XYgeff_fit(omega01_c_range)/2/np.pi/1e-3

patrial_Q1delta = [forward_diff(f = Q1delta_fit, x = omega)
                   for omega in omega01_c_range]              #导数的一次方
patrial_Q2delta = [forward_diff(f = Q2delta_fit, x = omega)
                   for omega in omega01_c_range]              #导数的一次方
prtial_XYgeff = [forward_diff(f = XYgeff_fit, x = omega)
                   for omega in omega01_c_range]              #导数的一次方
patrial_Q1delta_square = [p**2 for p in patrial_Q1delta]      #导数的平方
patrial_Q2delta_square = [p**2 for p in patrial_Q2delta]      #导数的平方
prtial_XYgeff_square = [p**2 for p in prtial_XYgeff]      #导数的平方
purcell_Q1leakage = Q1leakage_fit(omega01_c_range)            #由于态泄露引起的purcell效应


plt.figure()
plt.plot([(omega) for omega in geff_xy_range],
          patrial_Q1delta_square,) 
plt.plot([(omega) for omega in geff_xy_range],
          patrial_Q2delta_square,) 
plt.plot([(omega) for omega in geff_xy_range],
          prtial_XYgeff_square,) 

plt.yscale('log')

endtime=int(time.time())
print('Run-cell time is', endtime-starttime, 's\n')



geff_need_list = [-20,-0]
geff_index_list = [np.argmin(np.abs(geff_xy_range-g)) for g in geff_need_list]


"""计算coupler斜率"""
peak_eigens = QCQ.QCQ_eigens([0,0,0])
f01_peak_list = np.array(peak_eigens['omega01_list'])/2/np.pi    #顶点的f_01
f02_peak_list = np.array(peak_eigens['omega02_list'])/2/np.pi    #顶点的f_02
fah_peak_list = [f02-2*f01 for f01,f02 in zip(f01_peak_list,f02_peak_list)]  #顶点的非简谐
omega_c_need = [omega01_c_range[index] for index in geff_index_list]
k_phi_list = [flux_k_max_sim(f01_peak_list[index_c]*1e9, 
                              fah_peak_list[index_c]*1e9, 
                              omega_c*1e9/2/np.pi)[1]
              for omega_c in omega_c_need]   #coupler的磁通斜率 Hz/Phi_0
k_square_list = [patrial_Q1delta_square[index]*k_phi**2 
                 for index,k_phi in zip(geff_index_list,k_phi_list)]
# k_square_list = [k/k_square_list[2] for k in k_square_list]
k_purcell_list = [purcell_Q1leakage[index]*k_phi**2 
                  for index,k_phi in zip(geff_index_list,k_phi_list)]
# k_purcell_list = [k/k_purcell_list[2] for k in k_purcell_list]

#%%

QCQ_eigens_list = [QCQ.QCQ_eigens(phi_ex_list = [phi_ex_q1, phi_ex_c, phi_ex_q2])
                   for phi_ex_c in phi_ex_clist]  
Eenergy_sub_lists = [eigens['Eenergy_sub_list'] for eigens in QCQ_eigens_list]   
psi1_index_lists = [eigens['psi1_index_list'] for eigens in QCQ_eigens_list]
E_energies_lists = [eigens['E_energies'] for eigens in QCQ_eigens_list]

Eenergy_Q1sub_list, Eenergy_Csub_list, Eenergy_Q2sub_list = [list(row) for row in zip(*Eenergy_sub_lists)] 
omega01_Q1sub_list, omega01_Csub_list, omega01_Q2sub_list = [[E[1]-E[0] for E in list(row) ]
                                                             for row in zip(*Eenergy_sub_lists)]
omega01_Q1sub_list, omega01_Csub_list, omega01_Q2sub_list = [[E[1]-E[0] for E in list(row) ]
                                                             for row in zip(*Eenergy_sub_lists)]
omega01_lists = [[E[i] - E[0] for i in psi1_index] 
                 for psi1_index,E in zip(psi1_index_lists,E_energies_lists)]
        
psi1_lists = [eigens['psi1_list'] for eigens in QCQ_eigens_list]   #不同coupler磁通下的系统单激发态列表
psi1_tensor_lists = [eigens['psi1_tensor_list'] for eigens in QCQ_eigens_list]  #不同coupler磁通下的无耦合系统单激发态列表

psi100_list, psi010_list, psi001_list = [list(row) for row in zip(*psi1_lists)]    #不同coupler磁通下的耦合系统100，101，001列表
psi100_sub_list, psi010_sub_list, psi001_sub_list = [list(row) for row in zip(*psi1_tensor_lists)]

if np.round(omega01_Q1sub_list[0]/2/np.pi,5) == np.round(omega01_Q2sub_list[0]/2/np.pi,5) : 
    #在0.01MHZ的量级判断比特频率是否对齐
    """频率对齐情况下，根据能级劈裂计算耦合强度 delta = 2g 和关断点"""
    g_eff = [abs(omega01_list[-1]-omega01_list[0])/2 for omega01_list in omega01_lists]
    off_index = np.argmin(g_eff)
    
else:
    """
    两个比特频率不相等时，计算max(|<001|100>'|,|<100|001>'|)中的最小值来寻找关断点。
    """
    leacage_q2Toq1_list = [(psi_001_sub*psi_001_sub.dag()*psi_100*psi_100.dag()).tr()
                           for psi_001_sub,psi_100 in zip(psi001_sub_list,psi100_list)]
    leacage_q1Toq2_list = [(psi_100_sub*psi_100_sub.dag()*psi_001*psi_001.dag()).tr()
                           for psi_100_sub,psi_001 in zip(psi100_sub_list,psi001_list)]
    leacage_qmax_list = [np.max([leacage_q1,leacage_q2]) for leacage_q1,leacage_q2 in
                         zip(leacage_q2Toq1_list,leacage_q1Toq2_list)]
    off_index = np.argmin(leacage_qmax_list)
    
    # psi_100_off,psi_101_off,psi_001_off = psi1_lists[off_index]
    # H_sys_list = [QCQ.Coupling_H([phi_ex_q1, phi_ex_c,phi_ex_q2])
    #               for phi_ex_c in phi_ex_clist]
    # g_eff = [(psi_100_off.dag()*(H_sys['H_total'])*psi_001_off).tr()
    #          for H_sys in H_sys_list]
    """
    利用JC模型可得，态交叠大小 |<100|001>'|或|<001|100>'| ~ sin(theta/2), 
    其中 theta = arctan(2g/Delta), Delta为裸态下两个比特的频率差。
    利用上式近似计算耦合强度
    """
    leacage_list = [(l_q1toq2+l_q2toq1)/2 for l_q1toq2,l_q2toq1 in 
                    zip(leacage_q1Toq2_list,leacage_q2Toq1_list)]
    delta_Q1Q2 = abs(omega01_Q1sub_list[0]-omega01_Q2sub_list[0])
    g_eff = [np.tan(2*np.arcsin(l**0.5))*delta_Q1Q2/2 for l in leacage_list]
    
plt.figure()
# plt.plot([o_list[1]/2/np.pi for o_list in omega01_lists],
#          [g/1e-3/2/np.pi*2 for g in g_eff])
plt.plot([o_list[1]/2/np.pi for o_list in omega01_lists],
         [g/1e-3/2/np.pi*2 for g in g_eff1])
plt.plot([o_list[1]/2/np.pi for o_list in omega01_lists],
         [g/1e-3/2/np.pi*2 for g in g_eff])
# plt.plot([o_list[1]/2/np.pi for o_list in omega01_lists],
#           [(l**0.5)*delta_Q1Q2*1e3/2/np.pi*2 for l in leacage_list])
# plt.plot([o_list[1]/2/np.pi for o_list in omega01_lists],
#          [np.tan(2*np.arcsin(l**0.5))*delta_Q1Q2*1e3/2/np.pi for l in leacage_list])
#%%
"""
分别给出QCQ系统的本征态及对应不同subsystem的缀饰1态(|100>,|010>,|001>)和缀饰2态 
"""
phi_ex_list = [0,0,0]
Eenergy_sub_list,Estate_sub_list,E_energies,E_states = QCQ.H_eigenstates(phi_ex_list)

"""分别根据子系统的本征1态和2态，计算搜索QCQ系统本征态中与其布居交叠最大的视为其对应的缀饰态"""
# psi0_sub_list = [Es[0] for Es in Estate_sub_list]   #不同子系统的本征0态
# psi1_sub_list = [Es[1] for Es in Estate_sub_list]   #不同子系统的本征1态
# psi2_sub_list = [Es[2] for Es in Estate_sub_list]   #不同子系统的本征2态

# ptrace_index = [[0], [1,2], [3]]                    #不同子系统对应的partial trace 的序号
# P1_lists = [[(psi_1*psi_1.dag()*E_s.ptrace(ptrace_index[i])).tr() for E_s in E_states[:10]]
#            for i,psi_1 in enumerate(psi1_sub_list)]      #计算不同子系统本征1态与QCQ系统的布居交叠列表
# P2_lists = [[(psi_2*psi_2.dag()*E_s.ptrace(ptrace_index[i])).tr() for E_s in E_states[:10]]
#             for i,psi_2 in enumerate(psi2_sub_list)]      #计算不同子系统本征2态与QCQ系统的布居交叠列表

psi1_tensor_index = [[[j, int(i)] for j, i in enumerate(row)] for row in np.eye(3)]  #索引矩阵
psi1_tensor_list = [tensor(*[Estate_sub_list[index_j[0]][index_j[1]] for index_j in index_i]) 
                    for index_i in psi1_tensor_index]      #无耦合时QCQ系统的单激发1态列表
psi2_tensor_index = [[[j, int(i)] for j, i in enumerate(row)] for row in 2*np.eye(3)]   #索引矩阵
psi2_tensor_list = [tensor(*[Estate_sub_list[index_j[0]][index_j[1]] for index_j in index_i]) 
                    for index_i in psi2_tensor_index]      #无耦合时QCQ系统的单激发2态列表

#使用 psi_1*psi_1.dag()*E_s*E_s.dag() 比ket2dm(psi_1)*ket2dm(E_s) 计算速度要快
P1_lists = [[(psi_1*psi_1.dag()*E_s*E_s.dag()).tr() for E_s in E_states[:10]]    
           for psi_1 in psi1_tensor_list]      #计算不同子系统本征1态与QCQ系统的布居交叠列表
P2_lists = [[(psi_2*psi_2.dag()*E_s*E_s.dag()).tr() for E_s in E_states[:10]]
            for psi_2 in psi2_tensor_list]     #计算不同子系统本征2态与QCQ系统的布居交叠列表

#计算布居交叠最大值得到对应缀饰1态的index并避免重复索引(重复索引会出现在两个子系统能级一样的情况下)
psi1_index_list = []
for l_i, p_l in enumerate(P1_lists) :
    pl_need = [value if index not in psi1_index_list else -2 
               for index, value in enumerate(p_l)]
    psi1_index_list.append(np.argmax(pl_need))   
#计算布居交叠最大值得到对应缀饰2态的index并避免重复索引
psi2_index_list = []
for l_i, p_l in enumerate(P2_lists) :
    pl_need = [value if index not in psi2_index_list else -2 
               for index, value in enumerate(p_l)]
    psi2_index_list.append(np.argmax(pl_need))   
# psi1_index_list = [np.argmax(p_l) for p_l in P1_lists]   #计算布居交叠最大值得到对应缀饰1态的index
# psi2_index_list = [np.argmax(p_l) for p_l in P2_lists]   #计算布居交叠最大值得到对应缀饰2态的index

psi1_list = [E_states[index] for index in psi1_index_list]     #对应不同系统缀饰1态列表
psi2_list = [E_states[index] for index in psi2_index_list]     #对应不同系统缀饰2态列表
# psi11_list = []

"""计算QCQ系统的101态"""
psi_101_sub = tensor(Estate_sub_list[0][1],Estate_sub_list[1][0],Estate_sub_list[-1][1])
P101_list = [(psi_101_sub*psi_101_sub.dag()*E_s*E_s.dag()).tr() for E_s in E_states[:12]]
psi_101_index = np.argmax(P101_list)     #计算布居交叠最大值得到对应缀饰101态的index
psi_101 = E_states[psi_101_index]       #QCQ系统的101缀饰态