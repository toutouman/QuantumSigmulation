#%%
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from QubitHamiltonian import Transom_H

#%% 定义比特哈密顿参数

C_q = 85e-15
R_j = 8500
# E_c=0.22*2*np.pi  #GHZ
# E_j=17*2*np.pi #GHZ

Transmon1 = Transom_H(C_q = C_q, R_j = R_j)

psi_0 = Transmon1.psi_0
psi_1 = Transmon1.psi_1
psi_2 = Transmon1.psi_2
rho_0=psi_0*psi_0.dag()
rho_1=psi_1*psi_1.dag()
rho_2=psi_2*psi_2.dag()
H_0 = Transmon1.Hq_0
omega_01 = Transmon1.omega_01
sigma_z = ket2dm(psi_1)-ket2dm(psi_0)
sigma_z = Transmon1.a_d*Transmon1.a
print(r'Qubit f_01 = ' + rf'{np.around(Transmon1.omega_01/2/np.pi,3)} GHz' )
print(r'Qubit f_ah = ' + rf'{np.around(Transmon1.anhar/2/np.pi,3)} GHz' )
#%%
gamma_Z = 5e-3*2*np.pi

psi_t0 = (psi_1+psi_0).unit()
t_range = np.linspace(0,500,1001)
result = mesolve(H_0, psi_t0 , t_range, [np.sqrt(gamma_Z) * sigma_z], [])

P = []
for i,t in enumerate(t_range):
    rho_i = result.states[i]
    psi0_i = (np.exp(-1j*omega_01*t)*psi_1+psi_0).unit()
    p_i = (ket2dm(psi0_i)*rho_i).tr()
    P.append(p_i)

plt.figure()
plt.plot(t_range, P)
plt.plot(t_range,[0.5*np.exp(-t/318)+0.5 for t in t_range])

#%% 在Gauss波包驱动下比特随时间演化
starttime=int(time.time())

T_gate=20 #ns
t_nums=501
t_range=np.linspace(0,2*T_gate,t_nums)  #用两个pi/2门拼成一个pi门
drive_amp = 0.1715

H_t, drive_wave, drive_wave_drag = Transmon1.driveQ_gauss_H(drive_amp = drive_amp, 
                                                          omega_d = Transmon1.omega_01,T_gate = T_gate)
result=mesolve(H_t,psi_0,t_range,[] ,[rho_0,rho_1,rho_2])
P_0=result.expect[0]
P_1=result.expect[1]
P_2=result.expect[2]
plt.figure()
plt.plot(t_range,P_1, label = r'$P_1$')
plt.plot(t_range,P_0, label = r'$P_0$')
plt.plot(t_range,P_2, label = r'$P_2$')
plt.xlabel('t (ns)')
plt.ylabel('Qubits Population')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_range,[drive_wave(t_range[i])
                  for i in range(t_nums)],'b',label='Drive wave')
plt.plot(t_range,[drive_wave_drag(t_range[i])
                  for i in range(t_nums)],'r--',label='Drag wave')
plt.xlabel('t (ns)')
plt.ylabel('Drive Amp')
plt.legend()
plt.show()

endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')
#%% 在Cossin波包驱动下比特随时间演化
starttime=int(time.time())

T_gate=20 #ns
t_nums=501
t_range=np.linspace(0,2*T_gate,t_nums) #用两个pi/2门拼成一个pi门
drive_amp = 0.1289

H_t, drive_wave, drive_wave_drag = Transmon1.driveQ_cos_H(drive_amp = drive_amp, induct_drive = False,
                                                          omega_d = Transmon1.omega_01,T_gate = T_gate)

result=mesolve(H_t,psi_0,t_range,[] ,[rho_0,rho_1,rho_2])
P_0=result.expect[0]
P_1=result.expect[1]
P_2=result.expect[2]
plt.figure()
plt.plot(t_range,P_1, label = r'$P_1$')
plt.plot(t_range,P_0, label = r'$P_0$')
plt.plot(t_range,P_2, label = r'$P_2$')
plt.xlabel('t (ns)')
plt.ylabel('Qubits Population')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_range,[drive_wave(t_range[i])
                  for i in range(t_nums)],'b',label='Drive wave')
plt.plot(t_range,[drive_wave_drag(t_range[i])
                  for i in range(t_nums)],'r--',label='Drag wave')
plt.xlabel('t (ns)')
plt.ylabel('Drive Amp')
plt.legend()
plt.show()

endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')
#%% （Rabi 2D 模拟） 固定时间下比特演化 VS 驱动频率&驱动幅值 Cossin波包 
starttime=int(time.time())

T_gate=20 #ns
t_nums=501
t_range=np.linspace(0,2*T_gate,t_nums) #用两个pi/2门拼成一个pi门

Drive_amp_list=np.linspace(0,0.7,36)
F01_detune=np.linspace(-100e-3*2*np.pi,100e-3*2*np.pi,51)


P_0=[]
P_1=[]
P_2=[]

def multi_jobs_Rabi2D(k):

    P_0=[]
    P_1=[]
    P_2=[]
    omega_drive=Transmon1.omega_01+F01_detune[k]
    for i in range(len(Drive_amp_list)):
        drive_amp = Drive_amp_list[i]
        H_t = Transmon1.driveQ_cos_H(drive_amp = drive_amp, 
                                       omega_d = omega_drive,T_gate = T_gate)[0]
        result=mesolve(H_t,psi_0,t_range,[] ,[rho_0,rho_1,rho_2])
        P_0.append(result.expect[0][-1])
        P_1.append(result.expect[1][-1])
        P_2.append(result.expect[2][-1])
    return(P_0,P_1,P_2)

data = Parallel(n_jobs=12, verbose=3)(delayed(multi_jobs_Rabi2D)(k) for k in range(len(F01_detune)))
plt.figure()
X,Y=np.meshgrid(Drive_amp_list,F01_detune/2/np.pi)
plt.pcolor(Y,X,[data[i][1] for i in range(len(data) )],cmap='jet',label = 'Rabi 2D')
plt.clim(0,1)
plt.ylabel('Drive Amp')
plt.xlabel(r'($\omega_{d}$-$\omega_{01})/2\pi$ (GHz)')
plt.colorbar(label='P1')
plt.legend()


endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')