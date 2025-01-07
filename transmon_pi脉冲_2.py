# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:09:35 2021

@author: 馒头你个史剃磅
"""

#%%
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed, parallel_backend
from Quibit_Cal_Function import Hqubit_charge,Hn_charge,Cal_Qubits_Freq_Charge,Hqubit_charge

#%%
class Drive_wv:
    def __init__(self, 
             omega_01=5*2*np.pi, # GHz, the drive frequency 
             eta=-0.25*2*np.pi,   #GHz, the anharmonicity of qubits
             Omega=0.05*2*np.pi, # GHz, the drive strength
             T_gate=20,    # ns, time of the drive gate
             wv_class='Cossin',  # drive wave class, should be "Cossin" for "Gaussian"
             drag_alpha=0.5,  # drag alpha
            ):
        self.omega_01=omega_01
        self.eta=eta
        self.Omega=Omega
        self.T_gate=T_gate
        self.wv_class=wv_class
        self.drag_alpha=drag_alpha
    def wv_envelope(self,t):
        T_gate=self.T_gate
        if self.wv_class == "Gaussian":
            sigma_0=T_gate*0.15
            return(np.exp(-(t-T_gate/2)**2/(2*sigma_0**2))+
                   np.exp(-(t-3*T_gate/2)**2/(2*sigma_0**2)))
        else:
            return(0.5-0.5*np.cos(t/T_gate*2*np.pi))
    def drag_wv_envelope(self,t):
        T_gate=self.T_gate
        drag_alpha=self.drag_alpha
        eta=self.eta
        if self.wv_class == "Gaussian":
            sigma_0=T_gate*0.15
            return(-np.exp(-(t-T_gate/2)**2/(2*sigma_0**2))*(t-T_gate/2)/sigma_0**2*(drag_alpha/eta)-
                   np.exp(-(t-3*T_gate/2)**2/(2*sigma_0**2))*(t-3*T_gate/2)/sigma_0**2*(drag_alpha/eta))
        else:
            return(0.5*2*np.pi/T_gate*np.sin(t/T_gate*2*np.pi)*(drag_alpha/eta))
    def Omega_I(self,t,arg={'phi_0':0}):
        phi_0=arg['phi_0']
        s_t=self.wv_envelope(t)
        omega_t=s_t*self.Omega*np.sin(self.omega_01*(t+phi_0))
        return omega_t
    def Omega_Q(self,t,arg={'phi_0':0}):
        phi_0=arg['phi_0']
        s_dt=self.drag_wv_envelope(t)
        omega_t=s_dt*self.Omega*np.cos(self.omega_01*(t+phi_0))
        return omega_t

#%%

starttime=int(time.time())
E_C=0.25*2*np.pi  #GHZ
E_J=12*2*np.pi #GHZ
N=15
n_g=0.5
T_gate=20 #ns
t_nums=301
t_range=np.linspace(0,2*T_gate,t_nums)

H_0=Hqubit_charge(E_C,E_J,N,n_g)
E_list,E_states=H_0.eigenstates()

# E_states=H_0
psi_0=E_states[0]
psi_1=E_states[1]
psi_2=E_states[2]
rho_0=psi_0*psi_0.dag()
rho_1=psi_1*psi_1.dag()
rho_2=psi_2*psi_2.dag()
omega_01=E_list[1]-E_list[0]
omega_12=E_list[2]-E_list[1]
eta=omega_12-omega_01
# print(omega_01/2/np.pi)
Drive_wave=Drive_wv(omega_01=omega_01,eta=eta,T_gate=T_gate,wv_class = "Gaussian",Omega=0.195)
#%%
starttime=int(time.time())
H=[H_0,[Hn_charge(N),Drive_wave.Omega_I],[Hn_charge(N),Drive_wave.Omega_Q]]
arg={'phi_0':0}
result=mesolve(H,psi_0,t_range,[] ,[rho_0,rho_1,rho_2],args=arg)
P_0=result.expect[0]
P_1=result.expect[1]
P_2=result.expect[2]
plt.figure()
plt.plot(t_range,P_1)

plt.figure()
plt.plot(t_range,[Drive_wave.wv_envelope(t_range[i])
                  for i in range(t_nums)],'b',label='I')
plt.plot(t_range,[Drive_wave.drag_wv_envelope(t_range[i])
                  for i in range(t_nums)],'r--',label='Q')
plt.show()

endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')
#%%
starttime=int(time.time())
Omega_rabi=np.linspace(0,0.10*2*np.pi,31)
F01_detune=np.linspace(-100e-3*2*np.pi,100e-3*2*np.pi,51)


# def multi_jobs_Rabi2D(k):
P_0=[]
P_1=[]
P_2=[]
# for k in range(len(F01_detune)):
Drive_wv_lists=[[ Drive_wv(omega_01=omega_01+F01_detune[k],eta=eta,T_gate=T_gate,Omega=Omega_rabi[i])
                 for k in range(len(F01_detune))]
                for i in range(len(Omega_rabi))]
def multi_jobs_Rabi2D(k):

    # print(k)
    P_0=[]
    P_1=[]
    P_2=[]
    omega_drive=omega_01+F01_detune[k]
    for i in range(len(Omega_rabi)):
        # print(i)
        Drive_wave=Drive_wv(omega_01=omega_drive,eta=eta,T_gate=T_gate,Omega=Omega_rabi[i])
        H=[H_0,[Hn_charge(N),Drive_wave.Omega_I],[Hn_charge(N),Drive_wave.Omega_Q]]
        arg={'phi_0':0}
        result=mesolve(H,psi_0,t_range,[] ,[rho_0,rho_1,rho_2],args=arg)
        P_0.append(result.expect[0][-1])
        P_1.append(result.expect[1][-1])
        P_2.append(result.expect[2][-1])
    return(P_0,P_1,P_2)

data = Parallel(n_jobs=12, verbose=4)(delayed(multi_jobs_Rabi2D)(k) for k in range(len(F01_detune)))
plt.figure()
X,Y=np.meshgrid(Omega_rabi/2/np.pi,F01_detune)
plt.pcolor(Y,X,[data[i][1] for i in range(len(data) )],cmap='jet')
plt.clim(0,1)
plt.colorbar(label='P1')
plt.legend()

# plt.figure()
# plt.plot(Omega_rabi/2/np.pi,[P_1[i][-1] for i in range(len(Omega_rabi))])
#waveform envelope of signal
# plt.figure()
# plt.plot(t_range,[Drive_wave.wv_envelope(t_range[i])
#                   for i in range(t_nums)],'b',label='I')
# plt.plot(t_range,[Drive_wave.drag_wv_envelope(t_range[i])
#                   for i in range(t_nums)],'r--',label='Q')
# plt.show()
# plt.legend()

# print(P_1[-1])

endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')

# %%
