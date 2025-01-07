# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 11:09:54 2023

@author: mantoutou
"""
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from scipy import optimize

e=1.60217e-19
h=6.62607e-34
k_b= 1.3806505e-23
hbar=h/2/np.pi

class Transom_H:
    def __init__(self,
                 C_q,          #比特对地电容  F
                 R_j,          #比特结电阻 常温  Ohm
                 N_level = 10,  #计算比特哈密顿量时的能级空间大小
                 phi_ex = 0,     #外部磁通偏置  phi_0  磁通量子*2pi
                 **options):
        self.C_q = C_q
        self.Resistance = R_j
        self.E_c=e**2/(2*self.C_q)/hbar/1e9   #比特电容能 GHz
        self.E_j, self.I_c = self.func_R2EjIc(self.Resistance,Delta_Al = 180e-6) 
        self.N_level = N_level
        self.phi_ex = phi_ex
        # self.Resistance, self.I_c = self.func_Ej2RIc(self.E_j)
        self.options = options
        
        self.Ej_phi = self.E_j*np.cos(phi_ex)
        self.a_d = create(N_level)
        self.a = destroy(N_level)
        self.n_zpf = (self.Ej_phi/(32*self.E_c))**0.25
        self.phi_zpf = (2*self.E_c/self.Ej_phi)**0.25
        self.n = self.n_zpf*1j*(self.a_d-self.a)
        self.phi = self.phi_zpf*(self.a+self.a_d)
        self.Hq_0 = self.singleQ_H()
        
        self.q_states, self.q_omegas, self.anhar = self.qubit_eigens()
        self.omega_01, self.omega_12 = self.q_omegas[:2]
        self.psi_0, self.psi_1, self.psi_2 = self.q_states[:3]
        
    def func_Ej2RIc(self,
                  E_j,    #比特约瑟夫森能 GHz
                  Delta_Al = 180e-6, #结的超导能隙 eV
                  ):
        """
        输入E_j(Hz)和超导能级Delta_Al(默认180eV)计算结电阻(Ohm)和临界电流(A)
        """
        # Ic = np.pi*Delta_Al/2/self.Resistance
        # E_j = Ic*hbar/2/e/hbar
        R = np.pi*Delta_Al*hbar/(E_j*hbar*1e9)/4/e
        I_c = np.pi*Delta_Al/2/R
        return(R, I_c)
    def func_R2EjIc(self,
                    R,
                    Delta_Al = 180e-6, #结的超导能隙 eV
                    ):
        """
        输入结电阻R_j(Ohm)和超导能级Delta_Al(默认180eV)计算E_j(Hz)和临界电流(A)
        """
        I_c = np.pi*Delta_Al/2/R
        E_j = I_c*hbar/2/e/hbar/1e9
        # R = np.pi*Delta_Al*hbar/(E_j*hbar*1e9)/4/e
        # I_c = np.pi*Delta_Al/2/R
        return(E_j, I_c)
    
    def singleQ_H(self):
        """
        计算单比特静止哈密顿
        """
        n=self.n
        phi=self.phi
        Ej_phi = self.E_j*np.cos(self.phi_ex)
        E_c = self.E_c
        H_q=4*E_c*n**2+(Ej_phi*phi**2/2-Ej_phi/24*phi**4+Ej_phi/720*phi**6-
                        Ej_phi/40320*phi**8+Ej_phi/3628800*phi**10)
        return(H_q)
    
    def qubit_eigens(self):
        """ 
        计算单比特频率能级，本征态，非简谐等 GHZ
        """
        H_t = self.singleQ_H()
        [eigenenergys,eigenstates]=H_t.eigenstates()
        omega_list=[(eigenenergys[i+1]-eigenenergys[i]) 
                    for i in range(len(eigenenergys)-1)]
        anhar=((eigenenergys[2]+eigenenergys[0]-2*eigenenergys[1]))
        return(eigenstates,omega_list,anhar)
    
    def driveQ_H(self, drive_wv_func, induct_drive = False):
        """
        计算在电容(电感)驱动下的单比特哈密顿量, 
        其中驱动相是电荷数哈密顿n(电容驱动), 或磁通哈密顿phi(电感驱动)。 Note:并非Q或Phi
        返回 Qutip含时哈密顿量list
        """
        if induct_drive:
            H_drive = self.phi
        else:
            H_drive = self.n
        Hq_0 = self.Hq_0
        H_t=[Hq_0,[H_drive,drive_wv_func]]
        return(H_t)
    
    def driveQ_cos_H(self, drive_amp, omega_d, T_gate, induct_drive = False,
                     is_drag = True, drag_alpha = 0.5, phi_d = 0):
        """
        计算在cos包络下的比特驱动哈密顿
        返回 Qutip含时哈密顿量list, 驱动波形，Drag波形。
        """
        wv_envelope = lambda t: 0.5-0.5*np.cos(t/T_gate*2*np.pi)
        drive_wave = lambda t: wv_envelope(t)*drive_amp*np.sin(omega_d*(t+phi_d))
        
        if is_drag:
            wv_drag_envelope = lambda t: (0.5*2*np.pi/T_gate*np.sin(t/T_gate*2*np.pi)*
                                          (drag_alpha/self.anhar))
            drive_wave_drag = lambda t: wv_drag_envelope(t)*drive_amp*np.cos(omega_d*(t+phi_d))
            
            drive_wave_func = lambda t , args = None:  drive_wave(t) + drive_wave_drag(t)
        else:
            drive_wave_drag = lambda t: 0,
            drive_wave_func = lambda t , args = None: drive_wave(t)
            
        H_t = self.driveQ_H(drive_wave_func, induct_drive)
        return(H_t, drive_wave, drive_wave_drag)

    def driveQ_gauss_H(self, drive_amp, omega_d, T_gate, induct_drive = False,
                     sigma_coffe = 0.15, is_drag = True, drag_alpha = 0.5, phi_d = 0):
        """
        计算在gauss包络下的比特驱动哈密顿
        返回 Qutip含时哈密顿量list, 驱动波形，Drag波形。
        """
        sigma_0=T_gate*sigma_coffe
        wv_envelope = lambda t: (np.exp(-(t-T_gate/2)**2/(2*sigma_0**2))+
                                 np.exp(-(t-3*T_gate/2)**2/(2*sigma_0**2)))
        drive_wave = lambda t: wv_envelope(t)*drive_amp*np.sin(omega_d*(t+phi_d))
        
        if is_drag:
            wv_drag_envelope = lambda t: (-np.exp(-(t-T_gate/2)**2/(2*sigma_0**2))*
                                          (t-T_gate/2)/sigma_0**2*(drag_alpha/self.anhar)-
                                          np.exp(-(t-3*T_gate/2)**2/(2*sigma_0**2))*
                                          (t-3*T_gate/2)/sigma_0**2*(drag_alpha/self.anhar))
            drive_wave_drag = lambda t: wv_drag_envelope(t)*drive_amp*np.cos(omega_d*(t+phi_d))
            
            drive_wave_func = lambda t , args = None:  drive_wave(t) + drive_wave_drag(t)
        else:
            drive_wave_drag = lambda t: 0,
            drive_wave_func = lambda t , args = None: drive_wave(t)
            
        H_t = self.driveQ_H(drive_wave_func, induct_drive)
        return(H_t, drive_wave, drive_wave_drag)

class Float_Transmon_H(Transom_H):
    """
    计算浮地比特的哈密顿量
    """
    def __init__(self,
                 C_g1,          #比特一个pad对地电容  F
                 C_g2,          #比特另一个pad对地电容  F
                 C_12,         #比特两个pad的互容
                 R_j,          #比特结电阻 常温  Ohm
                 N_level = 10,  #计算比特哈密顿量时的能级空间大小
                 phi_ex = 0,     #外部磁通偏置  phi_0
                 **options):
        C_q = C_12 + C_g1*C_g2/(C_g1+C_g2)
        self.Ec_p = e**2*(4*C_12+C_g1+C_g2)/(2*(C_g1*C_g2+
                                                C_12*C_g1+C_12*C_g2))/hbar/1e9   #自由电子能量  GHz
        self.E_pm = -4*e**2*(C_g1-C_g2)/(2*(C_g1*C_g2+
                                            C_12*C_g1+C_12*C_g2))/hbar/1e9    #比特与自由电子耦合能量(E_pm*n_p*n_m) GHz
        self.C_12 = C_12
        self.C_g1 = C_g1
        self.C_g2 = C_g2
        Transom_H.__int__(
            self,
            C_q = C_q,     #比特等效电容
            R_j = R_j,          #比特结电阻 常温  Ohm
            N_level = N_level,  #计算比特哈密顿量时的能级空间大小
            phi_ex = phi_ex,     #外部磁通偏置  phi_0
            **options
            )
        
class GQubitReasonator(Transom_H):
    """
    接地比特与谐振腔耦合
    """
    def __init__(
            self,
            C_q,          #比特对地电容  F
            Rj_q,          #比特结电阻 常温  Ohm
            g_rq,          #比特和腔耦合强度  Hz
            omega_r,       #读取腔角频率    Hz
            phi_ex = 0,     #外部磁通偏置  phi_0  磁通量子*2pi
            N_level_q = 10,  # 计算比特哈密顿量时的能级空间大小
            N_level_r = 15,  # 计算腔哈密顿量时的能级空间大小
            **options,
            ):
        self.C_q = C_q
        self.N_level_q = N_level_q
        self.N_level_r = N_level_r
        self.omega_r = omega_r
        self.g_rq = g_rq
        self.phi_ex = phi_ex
        
        self.ad_q = create(N_level_q)
        self.a_q = destroy(N_level_q)
        self.ad_r = create(N_level_r)
        self.a_r = destroy(N_level_r)
        self.H_r = omega_r*self.ad_r*self.a_r
        
        Transom_H.__init__(
            self,
            C_q = C_q,
            R_j = Rj_q,
            N_level = N_level_q,
            phi_ex = phi_ex,
            **options)
        
        self.Hq_0 = self.singleQ_H()
        self.Hsys_0 = tensor(self.Hq_0,qeye(self.N_level_r))+tensor(qeye(self.N_level_q),self.H_r)\
                      -g_rq*tensor((self.ad_q-self.a_q),(self.ad_r-self.a_r))
    
    def driveQ_H(self, drive_wv_func, induct_drive = False):
        """
        计算在电容(电感)驱动下的单比特+谐振器哈密顿量, 
        其中驱动相是电荷数哈密顿n(电容驱动), 或磁通哈密顿phi(电感驱动)。 Note:并非Q或Phi
        返回 Qutip含时哈密顿量list
        """
        if induct_drive:
            H_drive = tensor(self.phi, qeye(self.N_level_r))
        else:
            H_drive = tensor(self.n, qeye(self.N_level_r))
        H_0 = self.Hsys_0
        H_t=[H_0,[H_drive,drive_wv_func]]
        return(H_t)
                


class QubitCoupling_base: 
    """
    计算多体系统电容耦合哈密顿量 (只考虑了电容耦合，即输入的电容矩阵，各子系统的约瑟夫森能是独立的)
    输入系统的电容矩阵C (Q·C^-1·Q中的 电容矩阵C)， Ej随磁通变化的函数列表 Ej_func_list
    """
    def __init__(
            self,
            C_Matrix,       #系统的电容矩阵C (Q·C^-1·Q中的 电容矩阵C)
            Ej_func_list,   #Ej随磁通变化的函数列表
            N_level = 8,    #每个子系统的大小
            ):
        self.N_level = N_level
        self.C_Matrix = C_Matrix
        self.C_inv = np.linalg.inv(self.C_Matrix)      # 电容逆矩阵C^1 (Q·C^-1·Q中的 电容逆矩阵 C^1)
        self.Ec_Matrix = self.C_inv*e**2/hbar/1e9/2     #系统哈密顿量Ec矩阵 (H_Ec = 4*Q·Ec·Q)
        self.Ej_func_list = Ej_func_list
        self.Qagent_nums = np.shape(self.C_Matrix)[0]   #量子子系统的数目
        self.Ec_list = [self.Ec_Matrix[i][i] for i in range(self.Qagent_nums)]  #不同子系统的 Ec
        self.a=destroy(self.N_level)
        self.a_d=create(self.N_level)
    def func_R2EjIc(self,
                    R,
                    Delta_Al = 180e-6, #结的超导能隙 eV
                    ):
        """
        输入结电阻R_j(Ohm)和超导能级Delta_Al(默认180eV)计算E_j(Hz)和临界电流(A)
        """
        I_c = np.pi*Delta_Al/2/R
        E_j = I_c*hbar/2/e/hbar/1e9
        # R = np.pi*Delta_Al*hbar/(E_j*hbar*1e9)/4/e
        # I_c = np.pi*Delta_Al/2/R
        return(E_j, I_c)
        
    def H_n_phi(self,
                phi_ex_list,    #不同子系统的外部磁通相位   磁通量子*2pi
                ):
        """
        计算子系统电荷哈密顿及相位哈密顿，
        返回不同子系统 E_j 列表， 子空间下的电荷哈密顿及相位哈密顿列表
        """
        Ej_list = [self.Ej_func_list[q](phi_ex_list[q]) for q in range(self.Qagent_nums)]
        
        n_zpf_list = [((E_j)/(32*E_c))**0.25 for E_c,E_j in zip(self.Ec_list, Ej_list)]
        phi_zpf_list = [(2*E_c/(E_j))**0.25 for E_c,E_j in zip(self.Ec_list, Ej_list)]
        
        Hn_SubSys_list = [-n_zpf*1j*(self.a-self.a_d) for n_zpf in n_zpf_list]         #不同子系统的电荷哈密顿
        Hphi_SubSys_list = [phi_zpf*(self.a+self.a_d) for phi_zpf in phi_zpf_list]   #不同子系统的相位哈密顿
        return(Ej_list, Hn_SubSys_list, Hphi_SubSys_list)
   
    def Coupling_H(self,
              phi_ex_list):
        """
        计算耦合系统哈密顿量，
        返回不同子空间下的子系统哈密顿量列表，耦合系统下的电荷哈密顿量及相位列表，以及总的耦合哈密顿量
        """
        
        Ej_list, Hn_SubSys_list, Hphi_SubSys_list = self.H_n_phi(phi_ex_list)
        
        """计算不同子空间下的子系统哈密顿量列表 H_SubSys_list"""
        HEj_SubSys_list = [-E_j*H_phi.cosm() for E_j,H_phi in zip(Ej_list,Hphi_SubSys_list)]
        # H_index_list = [[qeye(self.N_level)]*self.Qagent_nums](self.Qagent_nums)
        H_SubSys_list = [4*self.Ec_list[q_i]*Hn_SubSys_list[q_i]**2+HEj_SubSys_list[q_i]
                          for q_i in range(self.Qagent_nums)]             #子空间下的子系统哈密顿量列表
        
        """计算总的系统哈密顿量H_total"""
        Hn_list = [tensor(*([qeye(self.N_level)]*q_i+[Hn_SubSys_list[q_i]]
                            +[qeye(self.N_level)]*(self.Qagent_nums-q_i-1)) 
                          ) for q_i in range(self.Qagent_nums) ]           #耦合系统下的电荷哈密顿量列表
        
        Hphi_list = [tensor(*([qeye(self.N_level)]*q_i+[Hphi_SubSys_list[q_i]]
                            +[qeye(self.N_level)]*(self.Qagent_nums-q_i-1)) 
                          ) for q_i in range(self.Qagent_nums) ]           #耦合系统下的相位哈密顿量列表
        ECinv_n_matrix = [sum(list(map(lambda x: x[0]*x[1], zip(i,Hn_list)))) for i in self.Ec_Matrix]
        H_Ec = 4*sum(list(map(lambda y: y[0]*y[1], zip(Hn_list,ECinv_n_matrix))))  #耦合系统下的总电容哈密顿量
        HEj_list = [tensor(*([qeye(self.N_level)]*q_i+[HEj_SubSys_list[q_i]]
                            +[qeye(self.N_level)]*(self.Qagent_nums-q_i-1)) 
                          ) for q_i in range(self.Qagent_nums) ]           #耦合系统下的约瑟夫森哈密顿量列表
        H_Ej = sum(HEj_list)
        H_total = H_Ec+H_Ej                         #总耦合哈密顿量
        return({'H_SubSys_list': H_SubSys_list,
                'Hn_list': Hn_list,
                'Hphi_list': Hphi_list,
                'H_total': H_total,})
    
    def H_eigenenergies(self,
                 phi_ex_list):
        """
        计算哈密顿量本征能
        """
        Coupling_H = self.Coupling_H(phi_ex_list)
        H_SubSys_list = Coupling_H['H_SubSys_list']
        Hn_list = Coupling_H['Hn_list']
        Hphi_list = Coupling_H['Hphi_list']
        H_total = Coupling_H['H_total']
        
        Eigenenergies_sub_list = [H_sub.eigenenergies() for H_sub in H_SubSys_list]
        Eigenenergies = H_total.eigenenergies()
        
        return(Eigenenergies_sub_list,
               Eigenenergies,)
    
    def H_eigenstates(self,
                 phi_ex_list):
        """
        计算哈密顿量本征态
        """
        
        Coupling_H = self.Coupling_H(phi_ex_list)
        H_SubSys_list = Coupling_H['H_SubSys_list']
        Hn_list = Coupling_H['Hn_list']
        Hphi_list = Coupling_H['Hphi_list']
        H_total = Coupling_H['H_total']
        
        Eigenstates_sub_list = [H_sub.eigenstates() for H_sub in H_SubSys_list]
        Eenergy_sub_list,Estate_sub_list = [list(x) for x in zip(*Eigenstates_sub_list)]
        E_energies, E_states = H_total.eigenstates()
        
        return(Eenergy_sub_list,Estate_sub_list,
               E_energies,E_states)
    def H_sub_eigenstates(self,
                 phi_ex_list):
        """
        计算各个系统子空间的哈密顿量本征态
        """
        
        Coupling_H = self.Coupling_H(phi_ex_list)
        H_SubSys_list = Coupling_H['H_SubSys_list']
        Hn_list = Coupling_H['Hn_list']
        Hphi_list = Coupling_H['Hphi_list']
        H_total = Coupling_H['H_total']
        
        Eigenstates_sub_list = [H_sub.eigenstates() for H_sub in H_SubSys_list]
        Eenergy_sub_list,Estate_sub_list = [list(x) for x in zip(*Eigenstates_sub_list)]
        
        return(Eenergy_sub_list,Estate_sub_list,)
    
    def driveQ_H(self, q_index, phi_ex_list, 
                 drive_wv_func, induct_drive = False):
        """
        计算在指定比特电容(电感)驱动下的系统哈密顿量, 
        其中驱动相是电荷数哈密顿n(电容驱动), 或磁通哈密顿phi(电感驱动)。 Note:并非Q或Phi
        返回 Qutip含时哈密顿量list
        """
        Coupling_H = self.Coupling_H(phi_ex_list)
        Hn_list = Coupling_H['Hn_list']
        Hphi_list = Coupling_H['Hphi_list']
        if induct_drive:
            H_drive = Hphi_list[q_index]
        else:
            H_drive = Hn_list[q_index]
        H_total = Coupling_H['H_total']
        H_t=[H_total,[H_drive,drive_wv_func]]
        return(H_t)
    
    def driveQ_cos_H(self, q_index, phi_ex_list, 
                     drive_amp, omega_d, T_gate, induct_drive = False,
                     is_drag = True, drag_alpha = 0.5, phi_d = 0):
        """
        计算在指定比特电容(电感)双cos包络驱动下的的系统哈密顿量, 
        返回 Qutip含时哈密顿量list, 驱动波形，Drag波形。
        """
        wv_envelope = lambda t: 0.5-0.5*np.cos(t/T_gate*2*np.pi)
        drive_wave = lambda t: wv_envelope(t)*drive_amp*np.sin(omega_d*(t+phi_d))
        
        if is_drag:
            wv_drag_envelope = lambda t: (0.5*2*np.pi/T_gate*np.sin(t/T_gate*2*np.pi)*
                                          (drag_alpha/self.anhar))
            drive_wave_drag = lambda t: wv_drag_envelope(t)*drive_amp*np.cos(omega_d*(t+phi_d))
            
            drive_wave_func = lambda t , args = None:  drive_wave(t) + drive_wave_drag(t)
        else:
            drive_wave_drag = lambda t: 0,
            drive_wave_func = lambda t , args = None: drive_wave(t)
            
        H_t = self.driveQ_H(q_index, phi_ex_list, 
                            drive_wave_func, induct_drive)
        return(H_t, drive_wave, drive_wave_drag)

    def driveQ_gauss_H(self, q_index, phi_ex_list, 
                       drive_amp, omega_d, T_gate, induct_drive = False,
                       sigma_coffe = 0.15, is_drag = True, drag_alpha = 0.5, phi_d = 0):
        """
        计算在指定比特电容(电感)双gauss包络驱动下的的系统哈密顿量, 
        返回 Qutip含时哈密顿量list, 驱动波形，Drag波形。
        """
        sigma_0=T_gate*sigma_coffe 
        wv_envelope = lambda t: (np.exp(-(t-T_gate/2)**2/(2*sigma_0**2))+
                                 np.exp(-(t-3*T_gate/2)**2/(2*sigma_0**2)))
        drive_wave = lambda t: wv_envelope(t)*drive_amp*np.sin(omega_d*(t+phi_d))
        
        if is_drag:
            wv_drag_envelope = lambda t: (-np.exp(-(t-T_gate/2)**2/(2*sigma_0**2))*
                                          (t-T_gate/2)/sigma_0**2*(drag_alpha/self.anhar)-
                                          np.exp(-(t-3*T_gate/2)**2/(2*sigma_0**2))*
                                          (t-3*T_gate/2)/sigma_0**2*(drag_alpha/self.anhar))
            drive_wave_drag = lambda t: wv_drag_envelope(t)*drive_amp*np.cos(omega_d*(t+phi_d))
            
            drive_wave_func = lambda t , args = None:  drive_wave(t) + drive_wave_drag(t)
        else:
            drive_wave_drag = lambda t: 0,
            drive_wave_func = lambda t , args = None: drive_wave(t)
            
        H_t = self.driveQ_H(q_index, phi_ex_list,
                            drive_wave_func, induct_drive)
        return(H_t, drive_wave, drive_wave_drag)


class QubitDTCouplerQubit(QubitCoupling_base):
    """
    计算DoubleTransmon Coupler 的QCQ系统哈密顿量
    """
    def __init__(
            self,
            C_q1, C_q2, C_c1, C_c2, C_c, #比特和coupler自身电容 F
            C_qc1, C_qc2, C_12, #比特和coupler耦合电容 F
            Ej_q1,          #比特1的顶点约瑟夫森能      单位：GHz
            Ej_q2,          #比特2的顶点约瑟夫森能      单位：GHz
            Ej_c1,          #coupler 第一个结的约瑟夫森能      单位：GHz
            Ej_c2,          #coupler 第二个结的约瑟夫森能      单位：GHz
            Ej_c12,          #coupler 第三个结的约瑟夫森能      单位：GHz
            N_level = 8,    
            **options):
        self.Ej_q1 = Ej_q1; self.Ej_q2 = Ej_q2
        self.Ej_c1 = Ej_c1; self.Ej_c2 = Ej_c2; self.Ej_c12 = Ej_c12
        self.C_q1 = C_q1; self.C_q2 = C_q2; 
        self.C_c1 = C_c1; self.C_c2 = C_q2; self.C_c = C_c
        self.C_qc1 = C_qc1; self.C_qc2 = C_qc2; self.C_12 = C_12
        
        """计算系统的电容矩阵C (Q·C^-1·Q中的 电容矩阵C)"""
        Cs_q1 = C_q1+C_qc1+C_12
        Cs_q2 = C_q2+C_qc2+C_12
        Cs_c1 = C_c1+C_c+C_qc1
        Cs_c2 = C_c2+C_c+C_qc2
        C_Matrix = np.array([
            [Cs_q1, -C_qc1, 0, -C_12],
            [-C_qc1, Cs_c1, -C_c, 0],
            [0, -C_c, Cs_c2, -C_qc2],
            [-C_12, 0, -C_qc2, Cs_q2]
            ])
        
        """计算不同系统Ej 随外部磁通变化的函数列表"""
        Ej_q1_func = lambda phi:  Ej_q1*np.cos(phi)
        Ej_q2_func = lambda phi:  Ej_q2*np.cos(phi)
        Ej_c1_func = lambda phi:  Ej_c1+Ej_c12*np.cos(phi)
        Ej_c2_func = lambda phi:  Ej_c2+Ej_c12*np.cos(phi)
        Ej_func_list = [Ej_q1_func, Ej_c1_func,Ej_c2_func,Ej_q2_func]
        
        QubitCoupling_base.__init__(
            self,
            C_Matrix = C_Matrix,
            N_level = N_level,
            Ej_func_list = Ej_func_list,
            **options)
        
    """由于存在电感耦合，重新定义哈密顿量"""
    def Coupling_H(self,
              phi_ex_list):
        """
        计算耦合系统哈密顿量，
        返回不同子空间下的子系统哈密顿量列表，耦合系统下的电荷哈密顿量及相位列表，以及总的耦合哈密顿量
        """
        
        phi_ex_list_b =  [phi for phi in phi_ex_list]
        phi_ex_list_b.insert(2, phi_ex_list_b[1])
        # print(phi_ex_list_b)
        Ej_list, Hn_SubSys_list, Hphi_SubSys_list = self.H_n_phi(phi_ex_list_b)
        
        
        """计算coupler 子空间下的coupler哈密顿量"""
        Ec_Matrix_c = self.Ec_Matrix[1:-1,1:-1]  #coupler子空间下的Ec矩阵
        Hn_SubC_list = [tensor(Hn_SubSys_list[1], qeye(self.N_level)),
                        tensor(qeye(self.N_level), Hn_SubSys_list[2]),]   #Coupler系统空间下的电荷哈密顿列表
        Hphi_SubC_list = [tensor(Hphi_SubSys_list[1], qeye(self.N_level)),
                            tensor(qeye(self.N_level), Hphi_SubSys_list[2]),]  #Coupler系统空间下的相位哈密顿列表
        ECinv_n_matrix_c = [sum(list(map(lambda x: x[0]*x[1], zip(i,Hn_SubC_list)))) for i in Ec_Matrix_c]
        HEc_sub_c = 4*sum(list(map(lambda y: y[0]*y[1], zip(Hn_SubC_list,ECinv_n_matrix_c)))) #coupler子空间下的 Ec哈密顿
        HEj_sub_c = -self.Ej_c1*(Hphi_SubC_list[0]).cosm()-self.Ej_c2*(Hphi_SubC_list[1]).cosm()\
                    -self.Ej_c12*(np.cos(phi_ex_list[1])*(Hphi_SubC_list[0]-Hphi_SubC_list[1]).cosm()
                                 +np.sin(phi_ex_list[1])*(Hphi_SubC_list[0]-Hphi_SubC_list[1]).sinm())
        Hc_sub = HEc_sub_c+HEj_sub_c    #Coupler系统空间下的coupler哈密顿量
        # print(Hc_sub)
        """计算不同子空间下的子系统哈密顿量列表 H_SubSys_list"""
        HEj_SubSys_list = [ -Ej_list[0]*(Hphi_SubSys_list[0]).cosm(),
                            HEj_sub_c,
                            -Ej_list[-1]*(Hphi_SubSys_list[-1]).cosm()]
        
        HEc_SubSys_list = [4*self.Ec_list[0]*Hn_SubSys_list[0]**2,
                           HEc_sub_c,
                           4*self.Ec_list[-1]*Hn_SubSys_list[-1]**2,]
        
        H_SubSys_list = [H_Ec+H_Ej for H_Ec,H_Ej in 
                         zip(HEc_SubSys_list, HEj_SubSys_list)]           #子空间下的子系统哈密顿量列表
        
        """计算总的系统哈密顿量H_total"""
        Hn_list = [tensor(*([qeye(self.N_level)]*q_i+[Hn_SubSys_list[q_i]]
                            +[qeye(self.N_level)]*(self.Qagent_nums-q_i-1)) 
                          ) for q_i in range(self.Qagent_nums) ]           #耦合系统下的电荷哈密顿量列表
        
        Hphi_list = [tensor(*([qeye(self.N_level)]*q_i+[Hphi_SubSys_list[q_i]]
                            +[qeye(self.N_level)]*(self.Qagent_nums-q_i-1)) 
                          ) for q_i in range(self.Qagent_nums) ]           #耦合系统下的相位哈密顿量列表
        ECinv_n_matrix = [sum(list(map(lambda x: x[0]*x[1], zip(i,Hn_list)))) for i in self.Ec_Matrix]
        H_Ec = 4*sum(list(map(lambda y: y[0]*y[1], zip(Hn_list,ECinv_n_matrix))))  #耦合系统下的总电容哈密顿量
        
        HEj_list = [tensor(HEj_SubSys_list[0],qeye(self.N_level),qeye(self.N_level),qeye(self.N_level)),
                    tensor(qeye(self.N_level), HEj_SubSys_list[1], qeye(self.N_level)),
                    tensor(qeye(self.N_level),qeye(self.N_level),qeye(self.N_level),HEj_SubSys_list[-1])]   #耦合系统下的约瑟夫森哈密顿量列表
        H_Ej = sum(HEj_list)
        H_total = H_Ec+H_Ej                         #总耦合哈密顿量
        return({'H_SubSys_list': H_SubSys_list,
                'Hn_list': Hn_list,
                'Hphi_list': Hphi_list,
                'H_total': H_total,})

    def QCQ_eigens(
            self,
            phi_ex_list,):
        """
        分别给出QCQ系统的本征态及对应不同subsystem的缀饰1态(|100>,|010>,|001>)和缀饰2态 
        """
        Eenergy_sub_list,Estate_sub_list,E_energies,E_states = self.H_eigenstates(phi_ex_list)
        
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

        psi1_index_list = [np.argmax(p_l) for p_l in P1_lists]   #计算布居交叠最大值得到对应缀饰1态的index
        psi2_index_list = [np.argmax(p_l) for p_l in P2_lists]   #计算布居交叠最大值得到对应缀饰2态的index
        
        psi1_list = [E_states[index] for index in psi1_index_list]     #对应不同系统缀饰1态列表
        psi2_list = [E_states[index] for index in psi2_index_list]     #对应不同系统缀饰2态列表
        # psi11_list = []
        
        """计算QCQ系统的101态"""
        psi_101_sub = tensor(Estate_sub_list[0][1],Estate_sub_list[-1][1])
        P101_list = [(psi_101_sub*psi_101_sub.dag()*E_s.ptrace([0,3])).tr() for E_s in E_states[:12]]
        psi_101_index = np.argmax(P101_list)     #计算布居交叠最大值得到对应缀饰101态的index
        psi_101 = E_states[psi_101_index]       #QCQ系统的101缀饰态
        
        return({'psi1_index_list': psi1_index_list,
                'psi2_index_list': psi2_index_list,
                'psi1_list': psi1_list, 'psi2_list': psi2_list,
                'psi_101_index': psi_101_index, 
                'Eenergy_sub_list': Eenergy_sub_list,
                'Estate_sub_list': Estate_sub_list, 'psi_101': psi_101,
                'E_energies': E_energies, 'E_states': E_states,})

class GQubitGQubit(QubitCoupling_base):
    """
    计算接地比特-接地比特直接耦合的QQ系统哈密顿
    """
    def __init__(
            self,
            C_q1,
            C_q2,
            C_g,
            Rj_q1,
            Rj_q2,
            N_level = 8,
            **options):
        self.C_q1 = C_q1; self.C_q2 = C_q2; self.C_g = C_g
        self.Rj_q1 = Rj_q1; self.Rj_q2 = Rj_q2
        self.N_level = N_level
        
        self.Ej_q1, self.Ic_q1 = self.func_R2EjIc(Rj_q1)
        self.Ej_q2, self.Ic_q2 = self.func_R2EjIc(Rj_q2)
        
        """计算系统的电容矩阵C (Q·C^-1·Q中的 电容矩阵C)"""
        Cs_q1 = C_q1+C_g
        Cs_q2 = C_q2+C_g
        C_Matrix = np.array([
            [Cs_q1, -C_g,],
            [-C_g, Cs_q2],
            ])
        
        """计算不同系统Ej 随外部磁通变化的函数列表"""
        Ej_q1_func = lambda phi:  self.Ej_q1*np.cos(phi)
        Ej_q2_func = lambda phi:  self.Ej_q2*np.cos(phi)
        Ej_func_list = [Ej_q1_func,Ej_q2_func]
        
        QubitCoupling_base.__init__(
            self,
            C_Matrix = C_Matrix,
            N_level = N_level,
            Ej_func_list = Ej_func_list,
            **options)
    def QQ_eigens(
            self,
            phi_ex_list,):
        """
        分别给出QQ系统的本征态及对应不同subsystem的缀饰1态(|10>,|01>)和缀饰2态 
        """
        Eenergy_sub_list,Estate_sub_list,E_energies,E_states = self.H_eigenstates(phi_ex_list)
        Qagent_nums = self.Qagent_nums
        """分别根据子系统的本征1态和2态，计算搜索QQ系统本征态中与其布居交叠最大的视为其对应的缀饰态"""

        psi1_tensor_index = [[[j, int(i)] for j, i in enumerate(row)] for row in np.eye(Qagent_nums)]  #索引矩阵
        psi1_tensor_list = [tensor(*[Estate_sub_list[index_j[0]][index_j[1]] for index_j in index_i]) 
                            for index_i in psi1_tensor_index]      #无耦合时QCQ系统的单激发1态列表
        psi2_tensor_index = [[[j, int(i)] for j, i in enumerate(row)] for row in 2*np.eye(Qagent_nums)]   #索引矩阵
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
        
        """计算QQ系统的11态"""
        psi_11_sub = tensor(Estate_sub_list[0][1],Estate_sub_list[-1][1])
        P11_list = [(psi_11_sub*psi_11_sub.dag()*E_s*E_s.dag()).tr() for E_s in E_states[:12]]
        psi_11_index = np.argmax(P11_list)     #计算布居交叠最大值得到对应缀饰101态的index
        psi_11 = E_states[psi_11_index]       #QCQ系统的101缀饰态
        
        return({'psi1_index_list': psi1_index_list,
                'psi2_index_list': psi2_index_list,
                'psi1_list': psi1_list, 'psi2_list': psi2_list,
                'psi1_tensor_list': psi1_tensor_list,
                'psi2_tensor_list': psi2_tensor_list,
                'psi_11_index': psi_11_index, 
                'Eenergy_sub_list': Eenergy_sub_list,
                'Estate_sub_list': Estate_sub_list, 'psi_11': psi_11,
                'E_energies': E_energies, 'E_states': E_states,})
    
    
class GQubitGCouplerGQubit(QubitCoupling_base):
    """
    计算接地比特-接地Coupler-接地比特 的QCQ系统哈密顿量
    """
    def __init__(
            self,
            C_q1,
            C_q2,
            C_c,
            C_qc1,
            C_qc2,
            C_12,
            Rj_q1,
            Rj_q2,
            Rj_c,
            N_level = 8,    
            **options):
        self.C_q1 = C_q1; self.C_q2 = C_q2; self.C_c = C_c
        self.C_qc1 = C_qc1; self.C_qc2 = C_qc2; self.C_12 = C_12
        self.Rj_q1 = Rj_q1; self.Rj_q2 = Rj_q2; self.Rj_c = Rj_c
        self.N_level = N_level
        
        self.Ej_q1, self.Ic_q1 = self.func_R2EjIc(Rj_q1)
        self.Ej_q2, self.Ic_q2 = self.func_R2EjIc(Rj_q2)
        self.Ej_c, self.Ic_c = self.func_R2EjIc(Rj_c)
        
        """计算系统的电容矩阵C (Q·C^-1·Q中的 电容矩阵C)"""
        Cs_q1 = C_q1+C_qc1+C_12
        Cs_q2 = C_q2+C_qc2+C_12
        Cs_c = C_c+C_qc1+C_qc2
        C_Matrix = np.array([
            [Cs_q1, -C_qc1, -C_12],
            [-C_qc1, Cs_c, -C_qc2],
            [-C_12, -C_qc2, Cs_q2],
            ])
        
        """计算不同系统Ej 随外部磁通变化的函数列表"""
        Ej_q1_func = lambda phi:  self.Ej_q1*np.cos(phi)
        Ej_q2_func = lambda phi:  self.Ej_q2*np.cos(phi)
        Ej_c_func = lambda phi:  self.Ej_c*np.cos(phi)
        Ej_func_list = [Ej_q1_func, Ej_c_func,Ej_q2_func]
        
        QubitCoupling_base.__init__(
            self,
            C_Matrix = C_Matrix,
            N_level = N_level,
            Ej_func_list = Ej_func_list,
            **options)
    def QCQ_eigens(
            self,
            phi_ex_list,):
        """
        分别给出QCQ系统的本征态及对应不同subsystem的缀饰1态(|100>,|010>,|001>)和缀饰2态 
        """
        Eenergy_sub_list,Estate_sub_list,E_energies,E_states = self.H_eigenstates(phi_ex_list)
        
        """分别根据子系统的本征1态和2态，计算搜索QCQ系统本征态中与其布居交叠最大的视为其对应的缀饰态"""
        # psi0_sub_list = [Es[0] for Es in Estate_sub_list]   #不同子系统的本征0态
        # psi1_sub_list = [Es[1] for Es in Estate_sub_list]   #不同子系统的本征1态
        # psi2_sub_list = [Es[2] for Es in Estate_sub_list]   #不同子系统的本征2态

        # ptrace_index = [[0], [1,2], [3]]                    #不同子系统对应的partial trace 的序号
        # P1_lists = [[(psi_1*psi_1.dag()*E_s.ptrace(ptrace_index[i])).tr() for E_s in E_states[:10]]
        #            for i,psi_1 in enumerate(psi1_sub_list)]      #计算不同子系统本征1态与QCQ系统的布居交叠列表
        # P2_lists = [[(psi_2*psi_2.dag()*E_s.ptrace(ptrace_index[i])).tr() for E_s in E_states[:10]]
        #             for i,psi_2 in enumerate(psi2_sub_list)]      #计算不同子系统本征2态与QCQ系统的布居交叠列表
        Qagent_nums = self.Qagent_nums
        psi1_tensor_index = [[[j, int(i)] for j, i in enumerate(row)] for row in np.eye(Qagent_nums)]  #索引矩阵
        psi1_tensor_list = [tensor(*[Estate_sub_list[index_j[0]][index_j[1]] for index_j in index_i]) 
                            for index_i in psi1_tensor_index]      #无耦合时QCQ系统的单激发1态列表
        psi2_tensor_index = [[[j, int(i)] for j, i in enumerate(row)] for row in 2*np.eye(Qagent_nums)]   #索引矩阵
        psi2_tensor_list = [tensor(*[Estate_sub_list[index_j[0]][index_j[1]] for index_j in index_i]) 
                            for index_i in psi2_tensor_index]      #无耦合时QCQ系统的单激发2态列表
        
        #使用 psi_1*psi_1.dag()*E_s*E_s.dag() 比ket2dm(psi_1)*ket2dm(E_s) 计算速度要快
        P1_lists = [[(psi_1*psi_1.dag()*E_s*E_s.dag()).tr() for E_s in E_states[:10]]    
                   for psi_1 in psi1_tensor_list]      #计算不同子系统本征1态与QCQ系统的布居交叠列表
        P2_lists = [[(psi_2*psi_2.dag()*E_s*E_s.dag()).tr() for E_s in E_states[:40]]
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
        omega01_list = [E_energies[index]-E_energies[0] for index in psi1_index_list]    #omega_01列表
        omega02_list = [E_energies[index]-E_energies[0] for index in psi2_index_list]    #omega_02列表
        
        """计算QCQ系统的101态"""
        psi_101_sub = tensor(Estate_sub_list[0][1],Estate_sub_list[1][0],Estate_sub_list[-1][1])
        P101_list = [(psi_101_sub*psi_101_sub.dag()*E_s*E_s.dag()).tr() for E_s in E_states[:25]]
        psi_101_index = np.argmax(P101_list)     #计算布居交叠最大值得到对应缀饰101态的index
        psi_101 = E_states[psi_101_index]       #QCQ系统的101缀饰态
        
        return({'psi1_index_list': psi1_index_list,
                'psi2_index_list': psi2_index_list,
                'psi1_list': psi1_list, 'psi2_list': psi2_list,
                'psi1_tensor_list': psi1_tensor_list,
                'psi2_tensor_list': psi2_tensor_list,
                'omega01_list':omega01_list, 'omega02_list':omega02_list,
                'psi_101_index': psi_101_index, 
                'Eenergy_sub_list': Eenergy_sub_list,
                'Estate_sub_list': Estate_sub_list, 'psi_101': psi_101,
                'E_energies': E_energies, 'E_states': E_states,})
    
    def Cal_geff(self,
                 phi_ex_clist = np.linspace(0,np.pi*0.4,51),
                 phi_ex_q1 = 0,
                 phi_ex_q2 = 0):
        """
        计算比特-比特等效耦合强度及关断点
        """
        index_q1 = 0
        index_q2 = 2
        index_c = 1
        QCQ_eigens_list = [self.QCQ_eigens(phi_ex_list = [phi_ex_q1, phi_ex_c, phi_ex_q2])
                           for phi_ex_c in phi_ex_clist]  
        Eenergy_sub_lists = [eigens['Eenergy_sub_list'] for eigens in QCQ_eigens_list]   
        psi1_index_lists = [eigens['psi1_index_list'] for eigens in QCQ_eigens_list]
        E_energies_lists = [eigens['E_energies'] for eigens in QCQ_eigens_list]

        Eenergy_Q1sub_list, Eenergy_Csub_list, Eenergy_Q2sub_list = [list(row) for row in zip(*Eenergy_sub_lists)] 
        omega01_Q1sub_list, omega01_Csub_list, omega01_Q2sub_list = [[E[1]-E[0] for E in list(row) ]
                                                                     for row in zip(*Eenergy_sub_lists)]
        omega01_lists = [eigens['omega01_list'] for eigens in QCQ_eigens_list]
        omega02_lists = [eigens['omega02_list'] for eigens in QCQ_eigens_list]
        omega01_q1_list,omega01_c_list,omega01_q2_list = list(map(list, zip(*omega01_lists))) #行列互换得到各自的频率列表
                
        psi1_lists = [eigens['psi1_list'] for eigens in QCQ_eigens_list]   #不同coupler磁通下的系统单激发态列表
        psi1_tensor_lists = [eigens['psi1_tensor_list'] for eigens in QCQ_eigens_list]  #不同coupler磁通下的无耦合系统单激发态列表

        psi100_list, psi010_list, psi001_list = [list(row) for row in zip(*psi1_lists)]    #不同coupler磁通下的耦合系统100，101，001列表
        psi100_sub_list, psi010_sub_list, psi001_sub_list = [list(row) for row in zip(*psi1_tensor_lists)]
        
        omega101_index_list = [eigens['psi_101_index'] for eigens in QCQ_eigens_list]
        omega101_list = [E[i] - E[0] for i, E in zip(omega101_index_list, E_energies_lists)]
        
        leakage_cToq1_list = [(ket2dm(psi_100)*ket2dm(psi_010_sub)).tr()    #耦合系统Q1的1缀饰态含有coupler裸态|1>的概率
                             for psi_100,psi_010_sub in zip(psi100_list, psi010_sub_list)]
        leakage_cToq2_list = [(ket2dm(psi_001)*ket2dm(psi_010_sub)).tr()   #耦合系统Q2的1缀饰态含有coupler裸态|1>的概率
                             for psi_001,psi_010_sub in zip(psi001_list, psi010_sub_list)]
        """计算XY耦合强度及XY关断点"""
        if np.round(omega01_Q1sub_list[0],5) == np.round(omega01_Q2sub_list[0],5): 
            #在0.01*2*np.pi MHZ的量级判断比特频率是否对齐
            print('The F01s of Qubit1 and Qubit2 is equal! ')
            """频率对齐情况下，根据能级劈裂计算耦合强度 delta = 2g 和关断点"""
            g_xy_list = [(omega01_list[index_q2]-omega01_list[index_q1])/2 
                         for omega01_list in omega01_lists]
            off_xy_index = np.argmin(np.abs(g_xy_list))
            g_xy_list_leakage = []
            
            
        else:
            print('The F01s of Qubit1 and Qubit2 is not equal! ')
            """
            两个比特频率不相等时，计算max(|<001|100>'|,|<100|001>'|)中的最小值来寻找关断点。
            """
            leacage_q2Toq1_list = [(psi_001_sub*psi_001_sub.dag()*psi_100*psi_100.dag()).tr()
                                   for psi_001_sub,psi_100 in zip(psi001_sub_list,psi100_list)]
            leacage_q1Toq2_list = [(psi_100_sub*psi_100_sub.dag()*psi_001*psi_001.dag()).tr()
                                   for psi_100_sub,psi_001 in zip(psi100_sub_list,psi001_list)]
            leacage_qmax_list = [np.max([leacage_q1,leacage_q2]) for leacage_q1,leacage_q2 in
                                 zip(leacage_q2Toq1_list,leacage_q1Toq2_list)]
            off_xy_index = np.argmin(leacage_qmax_list)
            
            # psi_100_off,psi_101_off,psi_001_off = psi1_lists[off_index]
            # H_sys_list = [QCQ.Coupling_H([phi_ex_q1, phi_ex_c,phi_ex_q2])
            #               for phi_ex_c in phi_ex_clist]
            # g_eff = [(psi_100_off.dag()*(H_sys['H_total'])*psi_001_off).tr()
            #          for H_sys in H_sys_list]
            """
            利用JC模型可得，态交叠大小 |<100|001>'|或|<001|100>'| ~ sin(theta/2), 
            其中 theta = arctan(2g/Delta), Delta为裸态下两个比特的频率差。
            利用上式近似计算耦合强度，但是此法计算的g在关断点不为0，且无法判断正负向。
            """
            leacage_list = [(l_q1toq2+l_q2toq1)/2 for l_q1toq2,l_q2toq1 in 
                            zip(leacage_q1Toq2_list,leacage_q2Toq1_list)]  #计算平均泄露
            delta_Q1Q2 = abs(omega01_Q1sub_list[0]-omega01_Q2sub_list[0])
            g_xy_list_leakage = [np.tan(2*np.arcsin(abs(l)**0.5))*delta_Q1Q2/2 for l in leacage_list]
            
            
            """
            然后可以利用比特频率相对指定点(比如2M耦合位置)的偏移量来计算耦合强度，
            因为用态泄露计算的关断点不为0，且只有绝对值
            """
            g_ref_index = 0 if g_xy_list_leakage[0]>2e-3*2*np.pi else np.argmin(np.abs(np.abs(g_xy_list_leakage)-2e-3*2*np.pi))
            g_ref= g_xy_list_leakage[g_ref_index]
            
            delta01_q1_list = [ o-omega01_q1_list[g_ref_index] for o in omega01_q1_list]  #Q1频率与参考点的频率差
            delta01_q2_list = [ o-omega01_q2_list[g_ref_index] for o in omega01_q2_list]  #Q2频率与参考点的频率差
            g_delta_list = [(omega_q1+omega_q2)/2 for omega_q1,omega_q2 
                              in zip(delta01_q1_list,delta01_q2_list)]               #计算平均频率的偏移大小
            print(g_ref_index, g_ref/2/np.pi, g_delta_list[off_xy_index]/2/np.pi)
            g_ref = -1*g_ref if g_delta_list[off_xy_index]>0 else 1*g_ref    #判断参考点的耦合的正负向
            g_xy_list = [g+g_ref for g in g_delta_list]
        
        """计算ZZ耦合强度及ZZ关断点"""
        g_zz_list = [ omega_101-omega01_list[-1]-omega01_list[0] 
                     for omega01_list, omega_101 in zip(omega01_lists,omega101_list)]
        off_zz_index = np.argmin(np.abs(g_zz_list))
        
        return(
            {'g_xy_list': g_xy_list,
             'off_xy_index': off_xy_index,
             'omega01_lists': omega01_lists,
             'omega02_lists': omega02_lists,
             'g_zz_list': g_zz_list,
             'off_zz_index': off_zz_index,
             'QCQ_eigens_list': QCQ_eigens_list,
             'leakage_cToq1_list': leakage_cToq1_list,
             'leakage_cToq2_list': leakage_cToq2_list,
             'g_xy_list_leakage': g_xy_list_leakage,
             }
            )
        
class QubitCouplerQubit_H:
    """
    计算比特-coupler-比特耦合哈密顿量等，
    参考公式 H_t = 4Ec_q1*n_q1^2+4Ec_q2*n_q2^2+4Ec_c*n_c^2+
                  Ej_q1*cos(phi_q1)+Ej_q2*cos(phi_q2)+Ej_c*cos(phi_c)+
                  E_qc1*n_q1*n_c+E_qc2*n_c*n_q2+E_12*n_q1*n_q2
    """
    def __int__(
            self,
            Ec_q1,
            Ec_q2,
            Ec_c,
            Ej_q1,
            Ej_q2,
            Ej_c,
            E_qc1,
            E_qc2,
            E_q12,
            N_Qlevel = 10,  #计算比特哈密顿量时的能级空间大小
            N_Clevel = 10,  #计算coupler哈密顿量时的能级空间大小
            phi_ex_q1 = 0,     #外部磁通偏置  phi_0
            phi_ex_q2 = 0,     #外部磁通偏置  phi_0
            phi_ex_c = 0,     #外部磁通偏置  phi_0
            **options):
        self.Ej_q1 = Ej_q1; self.Ej_q2 = Ej_q2; self.Ej_c = Ej_c
        self.Ec_q1 = Ec_q1; self.Ec_q2 = Ec_q2; self.Ec_c = Ec_c
        self.E_qc1 = E_qc1; self.E_qc2 = E_qc2; self.E_q12 = E_q12
        self.N_Qlevel = N_Qlevel; self.N_Clevel = N_Clevel
        self.phi_ex_q1 = phi_ex_q1; self.phi_ex_q2 = phi_ex_q2; self.phi_ex_c = phi_ex_c
        self.options = options
        
        self.Ej_phi_q1 = Ej_q1*np.cos(phi_ex_q1)
        self.Ej_phi_q2 = Ej_q2*np.cos(phi_ex_q2)
        self.Ej_phi_c = Ej_c*np.cos(phi_ex_q2)
        self.n_zpf_q1 = (self.Ej_phi_q1/(32*self.Ec_q1))**0.25
        self.n_zpf_q2 = (self.Ej_phi_q2/(32*self.Ec_q2))**0.25
        self.n_zpf_c = (self.Ej_phi_c/(32*self.Ec_c))**0.25
        self.phi_zpf_q1 = (2*self.Ec_q1/self.Ej_phi_q1)**0.25
        self.phi_zpf_q2 = (2*self.Ec_q2/self.Ej_phi_q2)**0.25
        self.phi_zpf_c = (2*self.Ec_c/self.Ej_phi_c)**0.25
        # self.ad_q = create(N_Qlevel)
        
        self.Hq1_sub,self.Hq2_sub,self.Hc_sub,self.H_0,self.H_g,self.H_t = self.QCQ_H()
        self.H_q1 = tensor(self.Hq1_sub,qeye(self.N_Clevel),qeye(self.N_Qlevel))
        self.H_q2 = tensor(qeye(self.N_Qlevel),qeye(self.N_Clevel),self.Hq2_sub)
        self.H_c = tensor(qeye(self.N_Qlevel),self.Hc_sub,qeye(self.N_Qlevel))
        
        (self.Eenergy_q1_sub, self.Estate_q1_sub,
         self.Eenergy_q2_sub, self.Estate_q2_sub,
         self.Eenergy_c_sub, self.Estate_c_sub,
         self.E_energy, self.E_state) = self.H_eigens()
        
        
    def QCQ_H(self,):
        """
        计算比特-coupler-比特哈密顿量
        """
        a_q = create(self.N_Qlevel)     #比特产生算符
        ad_q = destroy(self.N_Qlevel)   #比特湮灭算符
        a_c = create(self.N_Clevel)     #coupler产生算符
        ad_c = destroy(self.N_Clevel)   #coupler湮灭算符
        N_Qlevel = self.N_Qlevel
        N_Clevel = self.N_Clevel
        
        n_q=self.n_zpf_q*-1j*(a_q-ad_q)    #Qubit电荷算符
        phi_q=self.phi_zpf_q*(a_q+ad_q)    #Qubit相位算符
        n_c=self.n_zpf_c*-1j*(a_c-ad_c)    #coupler电荷算符
        phi_c=self.phi_zpf_c*(a_c+ad_c)    #coupler相位算符

        H_q1=4*self.Ec_q1*n_q**2-self.Ej_q1*phi_q.cosm()   #Qubit1 哈密顿
        H_q2=4*self.Ec_q2*n_q**2-self.Ej_q2*phi_q.cosm()   #Qubit2 哈密顿
        H_c=4*self.Ec_c*n_c**2-self.Ej_c*phi_c.cosm()      #coupler 哈密顿
        
        H_0=tensor(H_q1,qeye(N_Clevel),qeye(N_Qlevel))+\
            tensor(qeye(N_Qlevel),qeye(N_Clevel),H_q2)+\
            tensor(qeye(N_Qlevel),H_c,qeye(N_Qlevel))                  #系统自由项
        H_g=self.E_qc1*tensor(n_q,n_c,qeye(N_Qlevel))+\
            self.E_qc1*tensor(qeye(N_Qlevel),n_c,n_q)+\
            self.E_q12*tensor(n_q,qeye(N_Clevel),n_q)            #哈密顿量耦合项
        H_t = H_0+H_g
        return(H_q1, H_q2, H_c,
               H_0, H_g, H_t)
    
    def H_eigens(self):
        """
        计算哈密顿量本征态级本征能
        """
        [Eenergy_q1_sub,Estate_q1_sub] = self.Hq1_sub.eigenstates()
        [Eenergy_q2_sub,Estate_q2_sub] = self.Hq2_sub.eigenstates()
        [Eenergy_c_sub,Estate_c_sub] = self.Hc_sub.eigenstates()
        [E_energy,E_state] = self.H_t.eigenstates()
        
        return(Eenergy_q1_sub, Estate_q1_sub,
               Eenergy_q2_sub, Estate_q2_sub,
               Eenergy_c_sub, Estate_c_sub,
               E_energy, E_state)
        
