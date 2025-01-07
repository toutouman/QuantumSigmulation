
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import optimize
from joblib import Parallel, delayed
starttime=int(time.time())

N=8

e=1.6021766208e-19
h=6.626070154e-34
h_bar=h/2/np.pi
C_c1=40e-15 #fF
C_c2=40e-15 #fF
C_12=4e-15 #fF
C_t=np.sqrt(C_c1*C_c2+C_c1*C_12+C_c2*C_12)
EC_c1=(C_c2+C_12)*e**2/(2*C_t**2)/h_bar/1e9
EC_c2=(C_c1+C_12)*e**2/(2*C_t**2)/h_bar/1e9
EC_12=(C_12)*e**2/(C_t**2)/h_bar/1e9
EJ_c1=13.91*2*np.pi #7GHZ
EJ_c2=21.42*2*np.pi #8.5GHZ
EJ_12=(EJ_c1+EJ_c2)/6

EJ_c2_list=np.linspace(20,24,41)*2*np.pi

C_q1=90e-15
C_q2=90e-15
EC_q1=e**2/(2*C_q1)/h_bar/1e9
EC_q2=e**2/(2*C_q2)/h_bar/1e9
EJ_q1=10.35*2*np.pi
EJ_q2=20.031*2*np.pi
g_qc1=200e-3*2*np.pi
g_qc2=200e-3*2*np.pi
phi0_e12=-0.17919
num_t=401
num_phi=26
Phi_e12r=np.linspace(0.0,0.25,num_phi)

def Phie12toe(phi_e12):
    phi_e12=phi_e12*np.pi*2
    phi_e=phi_e12+np.arcsin((EJ_12/EJ_c1)*np.sin(phi_e12))+np.arcsin((EJ_12/EJ_c2)*np.sin(phi_e12))
    return(phi_e/2/np.pi)

Phi_er=[Phie12toe(i) for i in Phi_e12r]

time_r=np.linspace(0,100,num_t)

a=destroy(N)
a_d=create(N)

n_zpf_q1=(EJ_q1/(32*EC_q1))**0.25
phi_zpf_q1=(2*EC_q1/EJ_q1)**0.25
n_q1=-n_zpf_q1*1j*(a-a_d)
phi_q1=phi_zpf_q1*(a+a_d)
n_zpf_q2=(EJ_q2/(32*EC_q2))**0.25
phi_zpf_q2=(2*EC_q2/EJ_q2)**0.25
n_q2=-n_zpf_q2*1j*(a-a_d)
phi_q2=phi_zpf_q2*(a+a_d)

H_q1=4*EC_q1*n_q1**2-EJ_q1*phi_q1.cosm()
H_q2=4*EC_q2*n_q2**2-EJ_q2*phi_q2.cosm()

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
    # H_phi12=EJ_12p*dphi_e12/EJ_c1p*(C_c1*4*EC_c1*n_c1/e+C_c1*2*EC_12/e*n_c2)-\
    #     EJ_12p*dphi_e12/EJ_c2p*(C_c2*4*EC_c2*n_c2/e+C_c2*2*EC_12/e*n_c1)
    H_J=-EJ_c1p*phi_c1.cosm()-EJ_c2p*phi_c2.cosm()-EJ_12p*(phi_c2-phi_c1).cosm()-\
        EJ_12*((1-phi_c1.cosm())*phi_c2.sinm()-(1-phi_c2.cosm())*phi_c1.sinm())*np.sin(phi_e12)
    H=H_C+H_J
    H_tensor=tensor(qeye(N),H,qeye(N))
    return({'H':H, 'tensor_H':H_tensor})

H_q1t=tensor(H_q1,qeye(N),qeye(N),qeye(N))
H_q2t=tensor(qeye(N),qeye(N),qeye(N),H_q2)
H_qc1=-g_qc1*tensor((a-a_d),(a-a_d),qeye(N),qeye(N))
H_qc2=-g_qc2*tensor(qeye(N),qeye(N),(a-a_d),(a-a_d))

Xi_list=[]
for i in range(num_phi):
    phi_e12=Phi_e12r[i]
    H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e12)['tensor_H']
    # [E_energy,E_state]=H_t.eigenstates()
    E_energy=H_t.eigenenergies()
    # [Eenergy_q1,Estate_q1]=H_q1.eigenstates()
    # [Eenergy_q2,Estate_q2]=H_q2.eigenstates()
    # [Eenergy_c,Estate_c]=H_ct(phi0_e12)['H'].eigenstates()
    # psi_00=tensor(Estate_q1[0],Estate_c[0],Estate_q2[0])
    # psi_10=tensor(Estate_q1[1],Estate_c[0],Estate_q2[0])
    # psi_01=tensor(Estate_q1[0],Estate_c[0],Estate_q2[1])
    # psi_11=tensor(Estate_q1[1],Estate_c[0],Estate_q2[1])
    # # psi_00=E_state[0]
    # psi_10=E_state[1]
    # psi_01=E_state[2]
    # psi_11=E_state[6]
    # rho_00=psi_00*psi_00.dag()
    # rho_10=psi_10*psi_10.dag()
    # rho_01=psi_01*psi_01.dag()
    # rho_11=psi_11*psi_11.dag()

    # epsilon_i=(rho_11*H_t).tr()+(rho_00*H_t).tr()-(rho_10*H_t).tr()-(rho_01*H_t).tr()
    xi_i=E_energy[6]+E_energy[0]-E_energy[1]-E_energy[2]
    Xi_list.append(xi_i/2/np.pi)

plt.figure()
plt.plot(Phi_er,[i*1e6 for i in Xi_list],label='ZZ coupling')
plt.xlabel(r'$\phi_e$')
plt.ylabel(r'$\xi_{zz}$ kHz')
plt.plot([np.min(Phi_er),np.max(Phi_er)],[0,0],'k--')
plt.legend()
#%%
H_t0=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi0_e12)['tensor_H']
[E_energy,E_state]=H_t0.eigenstates()
S_s=(E_state[0]).full()
for i in range(len(E_state)-1):
    S_s=np.column_stack((S_s,E_state[i+1].full()))
S_q=Qobj(S_s,dims=[[8,8,8,8],[8,8,8,8]])
psi_1=basis(N**4,1);psi_1.dims=[[8,8,8,8],[1]];psi_10=psi_1
psi_2=basis(N**4,2);psi_2.dims=[[8,8,8,8],[1]];psi_01=psi_2
psi_4=basis(N**4,4);psi_4.dims=[[8,8,8,8],[1]];psi_20=psi_4
psi_6=basis(N**4,6);psi_6.dims=[[8,8,8,8],[1]];psi_11=psi_6
psi_8=basis(N**4,8);psi_8.dims=[[8,8,8,8],[1]];psi_02=psi_8
rho_00=psi_00*psi_00.dag()
rho_10=psi_10*psi_10.dag()
rho_01=psi_01*psi_01.dag()
rho_11=psi_11*psi_11.dag()
rho_20=psi_20*psi_20.dag()
rho_02=psi_02*psi_02.dag()
Gcz_list1=[]
Gcz_list2=[]
for i in range(num_phi):
    phi_e12=Phi_e12r[i]
    H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e12)['tensor_H']
    g1_i=psi_20.dag()*S_q.dag()*H_t*S_q*psi_11
    g2_i=psi_02.dag()*S_q.dag()*H_t*S_q*psi_11
    # xi_i=E_energy[6]+E_energy[0]-E_energy[1]-E_energy[2]
    Gcz_list1.append(g1_i.full()[0][0]/2/np.pi*1e3)
    Gcz_list2.append(g2_i.full()[0][0]/2/np.pi*1e3)
plt.figure()
plt.plot(Phi_er,Gcz_list1,label=r'$g_{eff,|20\rangle,|11\rangle}$')
plt.plot([Phie12toe(phi0_e12),Phie12toe(phi0_e12)],[0,np.max(Gcz_list1)],'k--')
plt.xlabel(r'$\phi_e$')
plt.ylabel(r'$g_{eff}$ (MHz)')
plt.legend()
plt.figure()
plt.plot(Phi_er,Gcz_list2,label=r'$g_{eff,|02\rangle,|11\rangle}$')
plt.plot([Phie12toe(phi0_e12),Phie12toe(phi0_e12)],[0,np.min(Gcz_list2)],'k--')
plt.xlabel(r'$\phi_e$')
plt.ylabel(r'$g_{eff}$ (MHz)')
plt.legend()
#%%
def H_ct_2d(phi_ei,EJ_c2):
    EJ_12=(EJ_c1+EJ_c2)/10
    phi_e12=phi_ei*2*np.pi
    EJ_c1p=np.sqrt(EJ_c1**2-EJ_12**2*np.sin(phi_e12)**2)
    EJ_c2p=np.sqrt(EJ_c2**2-EJ_12**2*np.sin(phi_e12)**2)
    EJ_12p=EJ_12*np.cos(phi_e12)
    n_zpf_c1=(EJ_c1p/(32*EC_c1))**0.25
    n_zpf_c2=(EJ_c2p/(32*EC_c2))**0.25
    phi_zpf_c1=(2*EC_c1/EJ_c1p)**0.25
    phi_zpf_c2=(2*EC_c2/EJ_c2p)**0.25
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
    H=H_C+H_J
    H_tensor=tensor(qeye(N),H_C+H_J,qeye(N))
    return({'H':H, 'tensor_H':H_tensor})
def multy_job(k):
    EJ_c2=EJ_c2_list[k]
    Epsilon_list=[]
    for i in range(num_phi):
        phi_e12=Phi_e12r[i]
        H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct_2d(phi_e12,EJ_c2)['tensor_H']
        # [E_energy,E_state]=H_t.eigenstates()
        E_energy=H_t.eigenenergies()
        # [Eenergy_q1,Estate_q1]=H_q1.eigenstates()
        # [Eenergy_q2,Estate_q2]=H_q2.eigenstates()
        # psi_00=tensor(Estate_q1[0],Estate_c[0],Estate_q2[0])
        # psi_10=tensor(Estate_q1[1],Estate_c[0],Estate_q2[0])
        # psi_01=tensor(Estate_q1[0],Estate_c[0],Estate_q2[1])
        # psi_11=tensor(Estate_q1[1],Estate_c[0],Estate_q2[1])
        # psi_00=E_state[0]
        # psi_10=E_state[1]
        # psi_01=E_state[2]
        # psi_11=E_state[6]
        # rho_00=psi_00*psi_00.dag()
        # rho_10=psi_10*psi_10.dag()
        # rho_01=psi_01*psi_01.dag()
        # rho_11=psi_11*psi_11.dag()
    
        # epsilon_i=(rho_11*H_t).tr()+(rho_00*H_t).tr()-(rho_10*H_t).tr()-(rho_01*H_t).tr()
        epsilon_i=E_energy[6]+E_energy[0]-E_energy[1]-E_energy[2]
        Epsilon_list.append(epsilon_i/2/np.pi)
    return(Epsilon_list)
        
data = Parallel(n_jobs=7, verbose=2)(delayed(multy_job)(k) for k in range(len(EJ_c2_list)))
plt.figure()
X,Y=np.meshgrid(Phi_e12r,EJ_c2_list)
plt.pcolor(X,Y,data,cmap='seismic')
plt.clim(-10e-6*2*np.pi,10e-6*2*np.pi)
plt.colorbar()
#%%    
def multy_job(k):
    phi_e12=Phi_e12r[k]
    H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e12)['tensor_H']
    
    [Eenergy_q1,Estate_q1]=H_q1.eigenstates()
    [Eenergy_q2,Estate_q2]=H_q2.eigenstates()
    [Eenergy_c,Estate_c]=H_ct(phi_e12)['H'].eigenstates()
    # [E_energy,E_state]=H_t.eigenstates()
    
    psi_10=tensor(Estate_q1[1],Estate_c[0],Estate_q2[0])
    psi_01=tensor(Estate_q1[0],Estate_c[0],Estate_q2[1])
    # psi_00=E_state[0]
    # psi_10=E_state[1]
    # psi_01=E_state[2]
    rho_10=psi_10*psi_10.dag()
    # rho_10=tensor(Estate_q1[1]*Estate_q1[1].dag(),qeye(N),qeye(N),Estate_q2[0]*Estate_q2[0].dag())
    rho_01=psi_01*psi_01.dag()
    
    result=mesolve(H_t,psi_01,time_r,[],[rho_10])
    
    F_list=result.expect[0]
    # for i in range(num_t):
    #     psi_i=result.states[i]
    #     rho_i=psi_i*psi_i.dag()
    #     F_i=(rho_i*rho_10).tr()
    #     F_list.append(F_i)
    return(F_list)
data = Parallel(n_jobs=7, verbose=2)(delayed(multy_job)(k) for k in range(len(Phi_e12r)))
plt.figure()
X,Y=np.meshgrid(time_r,Phi_e12r)
plt.pcolor(Y,X,np.real(data),cmap='jet')
plt.clim(0,1)
plt.colorbar()

def fit_geff(t,g,A,B,phi):
    func=np.cos(2*np.pi*g*t+phi)*A+B
    return(func)
Geff_list=[]
bounds=((0,0.4,0.4,0),(1,0.6,0.6,2*np.pi))
for i in range(len(Phi_e12r)):
    index_i=np.where(np.real(data[i])>0.95)[0][0] if np.max(data[i])>0.95 else -1
    g_0=0.5/time_r[index_i]
    g_i,A_i,B_i,phi_i=optimize.curve_fit(fit_geff, time_r, np.real(data[i]),[g_0,0.5,0.5,np.pi],bounds=bounds,maxfev = 40000)[0]
    Geff_list.append(g_i*1e3)
    # plt.figure()
    # plt.plot(time_r, np.real(data[i]))
    # plt.plot(time_r, [fit_geff(i,g_i,A_i,B_i,phi_i) for i in time_r])
plt.figure()
plt.plot(Phi_e12r,np.abs(Geff_list),label=r'$C_{12}$=0 fF')
# plt.plot(Phi_e12r,np.abs(Geff_list1),label=r'$C_{12}$=2 fF')
plt.ylabel(r'$2g_{eff} $ (MHz)')
plt.xlabel(r'$\phi_e/\phi_0$')
plt.legend()
#%%
phi_e12=0.3
H_t=H_q1t+H_q2t+H_qc1+H_qc2+H_ct(phi_e12)['tensor_H']

[Eenergy_q1,Estate_q1]=H_q1.eigenstates()
[Eenergy_q2,Estate_q2]=H_q2.eigenstates()
[Eenergy_c,Estate_c]=H_ct(phi_e12)['H'].eigenstates()
# [E_energy,E_state]=H_t.eigenstates()

psi_10=tensor(Estate_q1[1],Estate_c[0],Estate_q2[0])
psi_01=tensor(Estate_q1[0],Estate_c[0],Estate_q2[1])
# psi_00=E_state[0]
# psi_10=E_state[1]
# psi_01=E_state[2]
rho_10=psi_10*psi_10.dag()
# rho_10=tensor(Estate_q1[1]*Estate_q1[1].dag(),qeye(N),qeye(N),Estate_q2[0]*Estate_q2[0].dag())
rho_01=psi_01*psi_01.dag()

result=mesolve(H_t,psi_01,time_r,[],[rho_10])

F_list=result.expect[0]
# F_list=[]
# for i in range(num_t):
#     psi_i=result.states[i]
#     rho_i=psi_i*psi_i.dag()
#     F_i=(rho_i*rho_10).tr()
#     F_list.append(F_i)
plt.figure()
plt.plot(time_r,F_list)


Omega_0=[]
Omega_1=[]
Omega_2=[]
Omega_3=[]
Omega_4=[]
for i in range(num_phi):
    phi_e=Phi_er[i]
    # E_energy=H_ct(phi_12e).eigenenergies()
    [E_energy,E_state]=H_ct(phi_e)['H'].eigenstates()
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
plt.plot([0,1],[4,4],'k--',label=r'$\omega_{10}$')
plt.plot([0,1],[5.5,5.5],'b--',label=r'$\omega_{01}$')
plt.plot([0,1],[9.5,9.5],'r--',label=r'$\omega_{11}$')
plt.plot([0,1],[7.74,7.74],'g--',label=r'$\omega_{20}$')
plt.plot([0,1],[10.76,10.76],'c--',label=r'$\omega_{02}$')
plt.legend()
plt.ylabel(r'Frequency (GHZ)')
plt.xlabel(r'$\phi_e$')
# plt.plot(Phi_er,Omega_4)
# plt.ylim([0,10])
endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')