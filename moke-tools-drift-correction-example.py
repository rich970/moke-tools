#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:33:13 2020

@author: richard
"""
# %%
from moke import moketools as mt
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
filename = 'hyst-loop-data'
# Importing data:
mk1 = mt.importMOKEdata(filename)

# %%
# Add some drift to the data...
curve = 4.3e-8
slope = 1.4e-5
offset = 2.3
noise = 1e-3*np.random.random_sample(size=len(mk1.data['index']))

mk1.data['kerrV'] = mk1.data['kerrV'] \
    + mk1.data['index']*slope \
    + curve*mk1.data['index']**2 \
    + noise

plt.figure(1)
plt.plot(mk1.data['field'], mk1.data['kerrV'])

mk1.fixdrift(b=2.4, data_window=10, d_step = 2, plot_figs=True)
