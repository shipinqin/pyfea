# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:59:31 2022

@author: shipi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# E_df = pd.DataFrame([columns=['xx', 'yy']])
# eps_df = pd.DataFrame([])
E_list, eps_list, eps_inc_list = [], [], []
theta0 = 0
for theta in np.linspace(0, 4*np.pi, 4*60, endpoint=True):  # Rotation
    theta_inc = theta - theta0
    Q = np.array([[np.cos(theta), np.sin(theta), 0],  # Rotation tensor
                  [-np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    
    E = 1/2*(Q.T@Q-np.eye(3))  # Lagrangian strain
    eps = 1/2*(Q+Q.T-2*np.eye(3))  # Infinitesimal strain
    
    E_list.append([theta*180/np.pi, E[0, 0], E[1, 1]])
    eps_list.append([theta*180/np.pi, eps[0, 0], eps[1, 1]])
    
    Q_inc = np.array([[np.cos(theta_inc), np.sin(theta_inc), 0],  # Rotation tensor (inc)
                  [-np.sin(theta_inc), np.cos(theta_inc), 0],
                  [0, 0, 1]])
    eps_inc = 1/2*(Q_inc+Q_inc.T-2*np.eye(3))
    if eps_inc_list:
       eps_inc_list.append([theta*180/np.pi, eps_inc_list[-1][1]+eps_inc[0, 0], eps_inc_list[-1][2]+eps_inc[1, 1]])
    else:
       eps_inc_list.append([theta*180/np.pi, eps_inc[0, 0], eps_inc[1, 1]])
    
    theta0=theta
    
E_df = pd.DataFrame(E_list, columns=['theta', 'strain_x', 'strain_y'])
eps_df = pd.DataFrame(eps_list, columns=['theta', 'strain_x', 'strain_y'])
eps_inc_df = pd.DataFrame(eps_inc_list, columns=['theta', 'strain_x', 'strain_y'])

var = ['strain_x', 'strain_y']

# fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
# for i, ax in enumerate(axs):
#     ax.plot(E_df['theta'], E_df[var[i]], 'k-', label='Lagrangian strain')
#     ax.plot(eps_df['theta'], eps_df[var[i]], 'r--', label='Infinitesimal strain')
#     ax.set_ylabel(var[i])

fig = plt.figure()

i=0  # plot strain x
ax11 = fig.add_subplot(221)
ax11.plot(E_df['theta'], E_df[var[i]], color='k', linestyle='-', label='Lagrangian strain')
ax11.plot(eps_df['theta'], eps_df[var[i]], color='r', linestyle='--', label='Infinitesimal strain')
ax11.plot(eps_inc_df['theta'], eps_inc_df[var[i]], color='blue', linestyle='--', label='Infinitesimal strain (inc)')
ax11.set_ylabel(var[i])

i=1  # plot strain y
ax21 = fig.add_subplot(223, sharex=ax11)
ax21.plot(E_df['theta'], E_df[var[i]], color='k', linestyle='-', label='Lagrangian strain')
ax21.plot(eps_df['theta'], eps_df[var[i]], color='r', linestyle='--', label='Infinitesimal strain')
ax21.plot(eps_inc_df['theta'], eps_inc_df[var[i]], color='blue', linestyle='--', label='Infinitesimal strain (inc)')
ax21.set_ylabel(var[i])
ax21.set_xlabel(r'$\theta$ ($\degree$)')

i=1  # plot small rotation range
ax12 = fig.add_subplot(122)
threshold = 30
E_df_small = E_df[E_df['theta']<=threshold]
eps_df_small = eps_df[eps_df['theta']<=threshold]
eps_inc_df_small = eps_inc_df[eps_inc_df['theta']<=threshold]
ax12.plot(E_df_small['theta'], E_df_small[var[i]], color='k', linestyle='-', label='Lagrangian strain')
ax12.plot(eps_df_small['theta'], eps_df_small[var[i]], color='r', linestyle='--', label='Infinitesimal strain')
ax12.plot(eps_inc_df_small['theta'], eps_inc_df_small[var[i]], color='blue', linestyle='--', label='Infinitesimal strain (inc)')
ax12.set_ylabel(var[i])
ax12.yaxis.tick_right()
ax12.yaxis.set_label_position("right")
ax12.set_xlabel(r'$\theta$ ($\degree$)')

ax12.legend()

fig.tight_layout()
fig.savefig('infinitesimal_strain_in_pure_rotation.png')