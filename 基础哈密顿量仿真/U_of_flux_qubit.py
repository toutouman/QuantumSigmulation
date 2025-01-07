# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 21:34:15 2020

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

num=201
E_C=1
E_J=100*E_C
alpha=0.8
f=0.6
d=0
# n_g=0.5

Phi_r=np.linspace(-2,2,num)

U_s=np.zeros((num,num))
for i in range(num):
    phi_1=Phi_r[i]*np.pi
    for j in range(num):
        phi_2=Phi_r[j]*np.pi
        U=E_J*(2+alpha-2*np.cos(phi_1)*np.cos(phi_2)
               -alpha*np.cos(2*np.pi*f+2*phi_1))
        U_s[i][j]=U

"""""""""""曲面图"""""""""""""""
figure = plt.figure()
ax = figure.add_subplot(1,1,1,projection='3d')
X,Y=np.meshgrid(Phi_r,Phi_r)
cm=plt.cm.get_cmap('jet')
ax.plot_surface(X,Y,U_s,cmap=cm)
ax.set_xlabel(r'$\phi_2/\pi$')
ax.set_ylabel(r'$\phi_1/\pi$')
# plt.legend()
plt.show()

"""""""""""等高图"""""""""""""""
plt.figure()
cm=plt.cm.get_cmap('Spectral_r')
plt.pcolor(X,Y,U_s,cmap=cm)
plt.contour(X,Y,U_s,15,colors='dimgrey')
plt.xlabel(r'$\phi_2/\pi$')
plt.ylabel(r'$\phi_1/\pi$')
# plt.legend()
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
print('total run time: ',endtime-starttime,'s')