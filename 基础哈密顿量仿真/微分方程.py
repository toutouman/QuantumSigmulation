# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 16:03:13 2021

@author: 馒头你个史剃磅
"""
from sympy import *
init_printing(use_latex='mathjax')
E = Function('E')
t = symbols('t')
eta = symbols('eta')
k = symbols('k')
g = symbols('g')
b=dsolve(Eq(E(t).diff(t, t)+1j*k*E(t).diff(t)+eta**2*E(t),0), E(t))
print(latex(b))