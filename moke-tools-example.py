#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:33:13 2020

@author: richard
"""

from moke import moketools as mt
import matplotlib.pyplot as plt

# %%
plt.close('all')
filename = 'hyst-loop-data'
# Importing data:
mk1 = mt.importMOKEdata(filename)

# Inspecting data:
mk1.head(n=4)
mk1.header
mk1.columns

# referencing data and columns
mk1.data[1:5]
mk1.data['index'][1:5]

# Adding columns
shifted_field = mk1.data['field'] + 50
mk1.addcolumn(shifted_field, 'shifted field', data_type=float)
mk1.columns
# Centering and normalising loops
mk1.centreloopnormalise()

# Plotting loops with matplotlib:
plt.plot(mk1.data['field'], mk1.data['normalised-Kerr'],
         label='centered-normalised')

plt.plot(mk1.data['shifted field'], mk1.data['normalised-Kerr'],
         label='shifted-centered-normalised')

# Finding magnetic properties:
Hc = mk1.findHc()
Rem = mk1.findrem()

# Center the field data and plot:
mk1.centrefield()
plt.plot(mk1.data['field'], mk1.data['normalised-Kerr'],
         label='field-centered-normalised')

plt.legend()
