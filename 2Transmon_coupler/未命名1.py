# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:15:24 2023

@author: mantoutou
"""

from sympy import *
init_printing(use_latex='mathjax')
# E = Function('E')
C_1,C_4, C_S1 ,C_S2 = symbols('C_1,C_4,C_S1,C_S2')
C_c1,C_c2,C_14 , C_c= symbols('C_c1,C_c2,C_14, Cc')
Phi_1,Phi_4,Phi_M,Phi_P = symbols('Phi_1,Phi_4,Phi_M,Phi_P')
# C_SP = C_03+C_02+C_13+C_12+C_34+C_24
# C_SM = C_03-C_02+C_13-C_12+C_34-C_24
C_M = Matrix([
    [C_1, -C_c1, 0, -C_14],
    [-C_c1, C_S1, -C_c, 0],
    [0, -C_c, C_S2, -C_c2],
    [-C_14, 0 , -C_c2, C_4]
    ])
