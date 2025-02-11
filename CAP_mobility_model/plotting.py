#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:12:06 2020
@author: shreyasdevaraju
"""
import numpy as np
import pandas as pd
from collections import namedtuple
import seaborn as sb
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3



def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length
    fig1 = plt.figure()
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Iterations")
    plt.title("Episode Length over Time")
    plt.savefig('imageSTATS/Episode_Lengths.png')
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)
        plt.pause(0.001)


    # Plot the episode reward over time
    fig2 = plt.figure()
    plt.plot(stats.avg_accumulated_rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Avg.Acc.Reward")
    plt.title("Episode Avg.Acc.Reward over Time" )
    plt.savefig('imageSTATS/Avg_Acc_Reward_per_Episode.png')
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)
        plt.pause(0.001)


    # Plot the episode reward over time
    fig3 = plt.figure()
    print(stats.avg_accumulated_rewards_per_episode)
    rewards_smoothed = pd.Series(stats.avg_accumulated_rewards_per_episode).rolling(smoothing_window, min_periods=smoothing_window).mean()
    print(rewards_smoothed)
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Avg.Acc.Reward (Smoothed)")
    plt.title("Episode Avg.Acc.Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig('imageSTATS/AvgAccReward_per_Episode_SMOOTHWINDOW10.png')
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)
        plt.pause(0.001)


    # MSE vs episode
    fig4 = plt.figure()
#    _smoothed = pd.Series(stats.avg_accumulated_rewards_per_episode).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(stats.MSE)
    plt.xlabel("Episode")
    plt.ylabel("MSE")
    plt.title("MSE vs episodes")
    plt.savefig('imageSTATS/MSE.png')
    if noshow:
        plt.close(fig4)
    else:
        plt.show(fig4)
        plt.pause(0.001)


    # Forbenius vs episode
    fig5 = plt.figure()
#    _smoothed = pd.Series(stats.avg_accumulated_rewards_per_episode).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(stats.frobeniusnormE)
    plt.xlabel("Episode")
    plt.ylabel("frobenius_norm_Error")
    plt.title("frobenius_norm_Error vs episodes")
    plt.savefig('imageSTATS/frobenius_norm_Error.png')
    if noshow:
        plt.close(fig5)
    else:
        plt.show(fig5)
        plt.pause(0.001)



#    # Plot time steps and episode number
#    fig3 = plt.figure()
#    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
#    plt.xlabel("Time Steps")
#    plt.ylabel("Episode")
#    plt.title("Episode per time step")
#    if noshow:
#        plt.close(fig3)
#    else:
#        plt.show(fig3)
#        plt.pause(0.001)
#        plt.savefig('Episode per time step.png')

#    return fig1, fig2, fig3

def plot_Qtable_stats(Q_table, Q_state_update_count, noshow=False):

    # Plot Frequency of State Visits
#    fig11 = plt.figure()
    fig11 = plt.figure()
#    plt.stem( np.arange(len(Q_state_update_count.flatten())) , Q_state_update_count.flatten() )

    data= Q_state_update_count.flatten()
    plt.hist(data, bins=[0,1,10,100,1000,10000,100000,1000000],rwidth=0.99  )

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([10**-1,10**5])
    plt.grid(True, which='minor' ,ls='-')

    plt.xlabel("State Visitation Frequency")
    plt.ylabel("No. of States")
    plt.title("Hist. of State Visitation Frequency")
    plt.savefig('imageSTATS/State_Visit_Freq.png')
    if noshow:
        plt.close(fig11)
    else:
        plt.show(fig11)
        plt.pause(0.001)

#return fig11

def plot_qtable_image(Q_table, Q_state_action_count, Q_state_update_count, ith_episode):
    fig12 = plt.figure(num=12,figsize=(20,10))
    plt.clf()
    fig12.add_subplot(1,3,3)
    data = Q_table.reshape(11**5,5)
    sb.heatmap(data, annot=True, fmt='.0f', linewidth=0.5, vmin=-200, vmax=-100)
#    c=plt.imshow(Q_table.reshape(32,5), vmin=0, vmax=-50)
    plt.ylabel('state')
    plt.xlabel('actions')
    plt.title("Q(s,a) -Ep"+str(ith_episode))

    fig12.add_subplot(1,3,2)
    data = Q_state_action_count.reshape(11**5,5)
    sb.heatmap(data, annot=True,linewidth=0.5)
#    c=plt.imshow(Q_table.reshape(32,5), vmin=0, vmax=-50)
    plt.ylabel('state')
    plt.xlabel('actions')
    plt.title("State-ActionFreq:N(s,a) -Ep"+str(ith_episode))


    fig12.add_subplot(1,3,1)
    data = Q_state_update_count.reshape(11**5,1)
    sb.heatmap(data, annot =True, fmt='.0f', linewidth=0.5, vmin=0, vmax=200000)
#    c=plt.imshow(Q_table.reshape(32,5), vmin=0, vmax=-50)
    plt.ylabel('state')
    plt.xlabel('freq')
    plt.title("StateFreq:N(s) -Ep"+str(ith_episode))

    plt.savefig('imageQT/img_{}.png'.format(ith_episode),format='png')

    plt.clf()



def plot_performance_plots(rstats,nAgents, save_path, noshow=False):

    # Plot Frequency of State Visits
    fig14 = plt.figure()
    for rows in range(rstats.coverage.shape[0]):
        plt.plot(rstats.coverage[rows])
        plt.xlabel("Time(s)*100")
        plt.ylabel("Coverage %")
        plt.title("Coverage ")
        plt.grid(True)
    plt.savefig(save_path+'Coverage.png')
    if noshow:
        plt.close(fig14)
    else:
        plt.show(fig14)
        plt.pause(0.001)

    fig15 = plt.figure()
    plt.plot(np.mean(rstats.coverage, axis=0))
    plt.ylabel('Coverage %')
    plt.xlabel('Time(s)*100')
    plt.grid(True)
    plt.title("Avg_Coverage")
    plt.savefig(save_path+'Average_coverage_plot.png')
    if noshow:
        plt.close(fig15)
    else:
        plt.show(fig15)
        plt.pause(0.001)

    fig16 = plt.figure()
    plt.boxplot(rstats.no_connected_comp)
    plt.ylabel('NCC')
    plt.ylim(0,10)
    plt.grid(True)
    plt.title("NCC ")
    plt.savefig(save_path+'NCC.png')
    plt.close(fig16)

    fig17 = plt.figure()
    plt.boxplot(rstats.avg_deg_conn)
    plt.ylabel('ANC')
    plt.ylim(0,nAgents/2)
    plt.grid(True)
    plt.title("ANC ")
    plt.savefig(save_path+'ANC.png')
    plt.close(fig17)

    fig18 = plt.figure()
    sb.heatmap(rstats.frequencymap, cmap="jet")#, vmin=0 ,vmax=100 )
    plt.ylabel('cell visitaions')
    # plt.grid(True)
    plt.title("Visitation Frequency Map n30s20")
    plt.savefig(save_path+'freq_map.png')
    plt.close(fig18)    



def plot_Connectivity_Histogram(connectivity_histogram, save_path, noshow=False):

    # Plot Frequency of State Visits
    fig19 = plt.figure()
    plt.bar([i for i in range(11)], connectivity_histogram)
    plt.ylabel("Frequency")
    plt.xlabel("Dst-wt-Connectivity at next availble waypoint")
    plt.yscale('log')
    plt.grid(True)
    plt.title("Frequency of Dst-wt-Connectivity at next availble waypoint")
    plt.savefig(save_path+'Dst-wt-Connectivity_Freq.png')
    if noshow:
        plt.close(fig19)
    else:
        plt.show(fig19)
        plt.pause(0.001)
