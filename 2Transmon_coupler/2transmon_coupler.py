
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
starttime=int(time.time())

N=15
e=1.6021766208e-19
h=6.626070154e-34
h_bar=h/2/np.pi
C_c1=100e-15 #fF
C_c2=100e-15 #fF
C_12=0e-15 #fF
C_t=np.sqrt(C_c1*C_c2+C_c1*C_12+C_c2*C_12)
EC_c1=(C_c2+C_12)*e**2/(2*C_t**2)/h_bar/1e9
EC_c2=(C_c1+C_12)*e**2/(2*C_t**2)/h_bar/1e9
EC_12=(2*C_12)*e**2/(2*C_t**2)/h_bar/1e9
EJ_c1=13*2*np.pi #GHZ
EJ_c2=13*2*np.pi
EJ_12=1.3*2*np.pi

phi_e=0
num=201
# g=1/np.sqrt(2*np.pi)/sigma_0*1.019 #0.0125*2*np.pi,1.019来自Omega的偏差

# T=24 #18结果更好
# t_range=np.linspace(0,T,num) #ns
# N_g=np.linspace(-2,2,num)
n_range=np.linspace(0,N,N+1)
# n_g=0
Phi_er=np.linspace(-0.5,0.5,num)

# Omega=(E_J/E_C/8)**0.25/np.sqrt(2)
# lambda_0=0.5/2.976
# t_0=T/2


H_cphi=1/2*(qdiags([1]*(N-1),1)+qdiags([1]*(N-1),-1))
H_sphi=-1j/2*(qdiags([1]*(N-1),1)-qdiags([1]*(N-1),-1))

def H_n():
    H_t=0
    for i in range(N):
        n=int(n_range[i])
        H_c=(n-int(N/2))*basis(N,n)*basis(N,n).dag()
        H_t=H_t+H_c
    return(H_t)

Hn_c1=tensor(H_n(),qeye(N))
Hn_c2=tensor(qeye(N),H_n())
H_cphi1=tensor(H_cphi,qeye(N))
H_cphi2=tensor(qeye(N),H_cphi)
H_sphi1=tensor(H_sphi,qeye(N))
H_sphi2=tensor(qeye(N),H_sphi)

def H_ct(phi_ei):
    phi_e=phi_ei*2*np.pi
    H_C=4*EC_c1*Hn_c1**2+4*EC_c2*Hn_c2**2+4*EC_12*Hn_c1*Hn_c2
    # H_J1=-EJ_c1*(np.exp(1j*C_12*C_c2/C_t**2*phi_e)*H_cphi1)
    # H_J2=-EJ_c2*(np.exp(-1j*C_12*C_c1/C_t**2*phi_e)*H_cphi2)
    H_J1=-EJ_c1*(np.cos(C_12*C_c2/C_t**2*phi_e)*H_cphi1-np.sin(C_12*C_c2/C_t**2*phi_e)*H_sphi1)
    H_J2=-EJ_c2*(np.cos(C_12*C_c1/C_t**2*phi_e)*H_cphi2+np.sin(C_12*C_c1/C_t**2*phi_e)*H_sphi2)
    H_J12=-EJ_12*(np.cos(C_c2*C_c1/C_t**2*phi_e)*(H_cphi1*H_cphi2+H_sphi1*H_sphi2)-np.sin(C_c2*C_c1/C_t**2*phi_e)*(H_sphi2*H_cphi1-H_cphi2*H_sphi1))

    H=H_C+H_J1+H_J2+H_J12
    return(H)

Omega_0=[]
Omega_1=[]
Omega_2=[]
Omega_3=[]
Omega_4=[]
for i in range(num):
    phi_e=Phi_er[i]
    # E_energy=H_ct(phi_e).eigenenergies()
    [E_energy,E_state]=H_ct(phi_e).eigenstates()
    E_0=E_energy[0]
    E_1=E_energy[1]
    E_2=E_energy[2]
    E_3=E_energy[3]
    E_4=E_energy[4]
    E_5=E_energy[5]
    psi_0=E_state[0]
    psi_1=E_state[1]
    psi_2=E_state[2]
    rho_0=psi_0*psi_0.dag()
    rho_1=psi_1*psi_1.dag()
    rho_2=psi_2*psi_2.dag()
    Omega_0.append((E_1-E_0)/2/np.pi)
    Omega_1.append((E_2-E_0)/2/np.pi)
    Omega_2.append((E_3-E_0)/2/np.pi)
    Omega_3.append((E_4-E_0)/2/np.pi)
    Omega_4.append((E_5-E_0)/2/np.pi)
    # omega_02=(omega_0+omega_1)/2
    # alpha=omega_0-omega_1

plt.figure()
plt.plot(Phi_er,Omega_0)
plt.plot(Phi_er,Omega_1)
plt.plot(Phi_er,Omega_2)
plt.plot(Phi_er,Omega_3)
# plt.plot(Phi_er,Omega_4)
# plt.ylim([0,10])
endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')

#%%
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
starttime=int(time.time())

N=8
e=1.6021766208e-19
h=6.626070154e-34
h_bar=h/2/np.pi
C_c1=100e-15 #fF
C_c2=100e-15 #fF
C_12=0e-15 #fF
C_t=np.sqrt(C_c1*C_c2+C_c1*C_12+C_c2*C_12)
EC_c1=(C_c2+C_12)*e**2/(2*C_t**2)/h_bar/1e9
EC_c2=(C_c1+C_12)*e**2/(2*C_t**2)/h_bar/1e9
EC_12=(C_12)*e**2/(C_t**2)/h_bar/1e9
EJ_c1=13*2*np.pi #GHZ
EJ_c2=13*2*np.pi
EJ_12=1.3*2*np.pi

phi_e=0
num=201
# g=1/np.sqrt(2*np.pi)/sigma_0*1.019 #0.0125*2*np.pi,1.019来自Omega的偏差

# T=24 #18结果更好
# t_range=np.linspace(0,T,num) #ns
# N_g=np.linspace(-2,2,num)
n_range=np.linspace(0,N,N+1)
# n_g=0
Phi_e12r=np.linspace(0,1,num)

def Phie12toe(phi_e12):
    phi_e12=phi_e12*np.pi*2
    phi_e=phi_e12+np.arcsin((EJ_12/EJ_c1)*np.sin(phi_e12))+np.arcsin((EJ_12/EJ_c2)*np.sin(phi_e12))
    return(phi_e/2/np.pi)

Phi_er=[Phie12toe(i) for i in Phi_e12r]

# Omega=(E_J/E_C/8)**0.25/np.sqrt(2)
# lambda_0=0.5/2.976
# t_0=T/2

a=destroy(N)
a_d=create(N)

def H_ct(phi_ei):
    phi_e12=phi_ei*2*np.pi
    EJ_c1p=np.sqrt(EJ_c1**2-EJ_12**2*np.sin(phi_e12)**2)
    EJ_c2p=np.sqrt(EJ_c2**2-EJ_12**2*np.sin(phi_e12)**2)
    EJ_12p=EJ_12*np.cos(phi_e12)
    n_zpf_c1=((EJ_c1p+EJ_12p)/(32*EC_c1))**0.25
    n_zpf_c2=((EJ_c2p+EJ_12p)/(32*EC_c2))**0.25
    phi_zpf_c1=(2*EC_c1/(EJ_c1p+EJ_12p))**0.25
    phi_zpf_c2=(2*EC_c2/(EJ_c2p+EJ_12p))**0.25
    n_c1=tensor(-n_zpf_c1*1j*(a-a_d),qeye(N))
    n_c2=tensor(qeye(N),-n_zpf_c2*1j*(a-a_d))
    phi_c1=tensor(phi_zpf_c1*(a+a_d),qeye(N))
    phi_c2=tensor(qeye(N),phi_zpf_c2*(a+a_d))
    dphi_e12=0
    
    H_C=4*EC_c1*n_c1**2+4*EC_c2*n_c2**2+4*EC_12*n_c1*n_c2
    H_phi12=EJ_12p*dphi_e12/EJ_c1p*(C_c1*4*EC_c1*n_c1/e+C_c1*2*EC_12/e*n_c2)-\
        EJ_12p*dphi_e12/EJ_c2p*(C_c2*4*EC_c2*n_c2/e+C_c2*2*EC_12/e*n_c1)
    H_J=-EJ_c1p*phi_c1.cosm()-EJ_c2p*phi_c2.cosm()-EJ_12p*(phi_c2-phi_c1).cosm()-\
        EJ_12*((1-phi_c1.cosm())*phi_c2.sinm()-(1-phi_c2.cosm())*phi_c1.sinm())*np.sin(phi_e12)

    H=H_C+H_J+H_phi12
    return(H)

Omega_0=[]
Omega_1=[]
Omega_2=[]
Omega_3=[]
Omega_4=[]
for i in range(num):
    phi_12e=Phi_e12r[i]
    # E_energy=H_ct(phi_12e).eigenenergies()
    [E_energy,E_state]=H_ct(phi_12e).eigenstates()
    E_0=E_energy[0]
    E_1=E_energy[1]
    E_2=E_energy[2]
    E_3=E_energy[3]
    E_4=E_energy[4]
    E_5=E_energy[5]
    psi_0=E_state[0]
    psi_1=E_state[1]
    psi_2=E_state[2]
    rho_0=psi_0*psi_0.dag()
    rho_1=psi_1*psi_1.dag()
    rho_2=psi_2*psi_2.dag()
    Omega_0.append((E_1-E_0)/2/np.pi)
    Omega_1.append((E_2-E_0)/2/np.pi)
    Omega_2.append((E_3-E_0)/2/np.pi)
    Omega_3.append((E_4-E_0)/2/np.pi)
    Omega_4.append((E_5-E_0)/2/np.pi)
    # omega_02=(omega_0+omega_1)/2
    # alpha=omega_0-omega_1

# plt.figure()
plt.plot(Phi_er,Omega_0)
plt.plot(Phi_er,Omega_1)
plt.plot(Phi_er,Omega_2)
plt.plot(Phi_er,Omega_3)
# plt.plot(Phi_er,Omega_4)
# plt.ylim([0,10])
endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')



