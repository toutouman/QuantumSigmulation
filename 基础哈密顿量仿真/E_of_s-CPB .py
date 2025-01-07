# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 18:52:15 2020

@author: 馒头你个史剃磅
"""


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
starttime=int(time.time())
N=30
E_C=1
E_J=1*E_C
d=0
# n_g=0.5

step_N=49
N_g=np.linspace(-0.1,1.1,step_N)
Delta_r=np.linspace(-1.2,1.2,step_N)

n_range=np.linspace(0,N,N+1)

def H(n_g,delta):
    H_j=-E_J/2*((np.cos(delta/2)-1j*d*np.sin(delta/2))*qdiags([1]*(N-1),1)
                +(np.cos(delta/2)+1j*d*np.sin(delta/2))*qdiags([1]*(N-1),-1))
    H_t=H_j
    for i in range(N):
        n=int(n_range[i])
        H_c=E_C*(n-n_g-(N/2))**2*basis(N,n)*basis(N,n).dag()
        H_t=H_t+H_c
    return(H_t)

def multy_job(k):
    n_g=N_g[k]
    E_i=[]
    for i in range(step_N):
        delta=Delta_r[i]*np.pi
        H_t=H(n_g,delta)
        E_s=H_t.eigenenergies()
        e_i=[E_s[i]/E_C for i in range(4)]
        E_i.append(e_i)
    return(E_i)
data = Parallel(n_jobs=3, verbose=2)(delayed(multy_job)(k) for k in range(step_N))

E_0=[[data[i][j][0].real for i in range(step_N)] for j in range(step_N)]
E_1=[[data[i][j][1].real for i in range(step_N)] for j in range(step_N)]
E_2=[[data[i][j][2].real for i in range(step_N)] for j in range(step_N)]
E_3=[[data[i][j][3].real for i in range(step_N)] for j in range(step_N)]

figure = plt.figure()
ax = Axes3D(figure)
X,Y=np.meshgrid(N_g,Delta_r)
cm=plt.cm.get_cmap('plasma')
norm =mpl.colors.Normalize(vmin=np.min(E_0), vmax=np.max(E_1))
ax.plot_surface(X,Y,np.array(E_0),cmap=cm,norm=norm)
ax.plot_surface(X,Y,np.array(E_1),cmap=cm,norm=norm)
# ax.plot_surface(X,Y,np.array(E_2),cmap=cm,norm=norm)
# ax.plot_surface(X,Y,np.array(E_3),cmap=cm,norm=norm)
ax.set_xlabel(r'$n_g$')
ax.set_ylabel(r'$\delta/\pi$')
plt.show()

E_0bar=[E_0[i][int((step_N-1)/2)] for i in range(step_N)]
E_1bar=[E_1[i][int((step_N-1)/2)] for i in range(step_N)]
plt.figure()
plt.plot(Delta_r,E_0bar,label=r'$E_0(\delta)$ at $n_g=0.5$')
plt.plot(Delta_r,E_1bar,label=r'$E_1(\delta)$ at $n_g=0.5$')
plt.show()
# E_0=[i[0] for i in data]
# E_1=[i[1] for i in data]
# E_2=[i[2] for i in data]
# E_3=[i[3] for i in data]

# plt.figure()
# plt.plot(N_g,E_0,label=r'$E_0$')
# plt.plot(N_g,E_1,label=r'$E_1$')
# plt.plot(N_g,E_2,label=r'$E_2$')
# plt.plot(N_g,E_3,label=r'$E_3$')
# plt.legend()



endtime=int(time.time())
print('total run time is', endtime-starttime, 's\n')
