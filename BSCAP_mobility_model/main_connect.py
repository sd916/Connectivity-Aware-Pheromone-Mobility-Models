#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 8 2023
Modified on Feb 10 2025
@author: Shreyas Devaraju (devaraju@uci.edu), Aaditya Timalsina <aadityatimal@gmail.com>
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Modified for BS-Connectivity-Oriented Pheromone based Movement simulation
# Modified on Feb 10 2025 (Author: Shreyas Devaraju <devaraju@uci.edu>)
# Modified on Feb 10 2025, for multiple base station - Aaditya Timalsina <aadityatimal@gmail.com>
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from math import floor
import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'   # (or whatever value you like)

import sys
from scipy.io import savemat
from scipy.io import loadmat
import random

# import matplotlib
# matplotlib.use('Agg') #For not plot image at each episode
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

# Updated, swarm object version
import numpy as np
from numpy import asarray
from numpy import save

import simpy
# from tqdm import tqdm

import pheromone as ph
import time
import pickle
from collections import defaultdict
#______________________________________SD-new

import plotting
import globals # contains the Qtable defined as global variable 

STATS_INTERVAL=1  # STATS_INTERVAL=100 # log results interval (s)

running_as = ''

# main -start
# -------------------------------------------------------------------------------------------------------------------------------------------
def main():
    globals.init() # iniitializing global defined in globals.py 
    num_episodes = 20 # no. of runs for averaging - 30 runs
    nAgents = int(sys.argv[1]) # no. of UAVs; 50, 100 
    uav_speed = int(sys.argv[2]) # 20 m/s or 50 m/s
    transmission_range = int(sys.argv[3]) # 1200 m
    map_size = int(sys.argv[4])  # 8000m (gives 8km x 8km area)
    beta_type = float(sys.argv[5]) # tuning parameter; set beta to {0.5, 1.5, 2.5} for BSCAP model; 
                                    # (1.5 - mid connectiviy and coverage performance);(0.5 - low neighbor connectivity, high coverage);(2.5 - high neighbor connectivity, low coverage)
    MOBILILTY_MODEL_TYPE = 'BSCAP' #sys.argv[6]
    nBaseStations = 1 # Single base station node at mid bottom of map.
    simTime = 2000+5 # seconds; end time
    position_log_interval  = 205 # seconds; starts log positions at (simTime - position_log_interval); Hence we log the last position_log_interval seconds of node positions
    ca= 200 #int(sys.argv[8])  #collision_buffer; 200m

    # SAVED RESULT PATH
    save_path = f'{MOBILILTY_MODEL_TYPE}_{running_as}results_{num_episodes}runs/{nAgents}n-{uav_speed}mps-' + \
                    f'{transmission_range}tx-{map_size}map-{beta_type}beta-{MOBILILTY_MODEL_TYPE}model/' # output path
    if not os.path.exists(save_path): os.makedirs(save_path)
    print("(UAVs, speed, transmission_range, map_size, beta_type) = ", nAgents, uav_speed, transmission_range, map_size, beta_type)
    print(save_path)

    # SAVED Node Trajectories Path
    TRAJECTORY_FOLDER = f"./nodeTrajectories-{beta_type}beta/{MOBILILTY_MODEL_TYPE}/{uav_speed}mps_{nAgents}nodes/"; os.makedirs(TRAJECTORY_FOLDER, exist_ok=True) # Stores node trajectories
    # TRAJECTORY_FOLDER = f"./nodeTrajectories/{MOBILILTY_MODEL_TYPE}/{uav_speed}mps_{nAgents}nodes/"; os.makedirs(TRAJECTORY_FOLDER, exist_ok=True) # Stores node trajectories
    
    
    #SD: choose evaporation and diffusion rate as input
    evp=  0.006
    diff= 0.006
  
    # initialize statistics object 
    rstats = R_Stats(num_episodes, simTime)
# ________________________________________________________________________ 
# for loop every episode
    # For every episode
    for ith_episode in range(num_episodes):
        globals.no_episode = ith_episode # global episode counter
        print(f"Starting episode {ith_episode}")
        random_seed = ith_episode*100 + ith_episode
        set_random_seed(random_seed); print("Seed:", random_seed ) #SD*: Added np.random.seed(random_seed)
        
        # To plot network topology; set draw to True
        draw = False 

        env = simpy.Environment()
        # Reset and re-run simulation
        swarm = ph.uav_swarm(env, evaporation_rate = evp, diffusion_rate = diff, use_connect=True, # initaialize simulation environment
                        nAgents=nAgents, nBaseStations=nBaseStations, map_size=map_size, collision_avoidance=True, collision_buffer=float(ca),
                        stats_interval=STATS_INTERVAL,fwd_scheme=5, hop_dist=2, uav_speed=uav_speed,
                        map_resolution=100, transmission_range=transmission_range, alpha_type=beta_type)
        env.process(swarm.sim_start_3d_simpy(simTime, drawUAVs=draw, drawMap=False, plotInterval=10))  # rum simulation
        env.run(until=simTime)
        
        res = swarm.stats

        #save BSCAP node trajectories
        save_NodeTrajectories(TRAJECTORY_FOLDER, MOBILILTY_MODEL_TYPE, ith_episode, nAgents, res.UAV_positions, res.UAV_nextwaypoints, simTime-position_log_interval, simTime ) #SD*: saving node trajectories for 200s
        
        #Update stats and result for the episode
        rstats.coverage = np.append(rstats.coverage, [res.coverage],axis=0)
        rstats.no_connected_comp = np.append(rstats.no_connected_comp, np.mean(res.no_connected_comp))
        rstats.avg_deg_conn = np.append(rstats.avg_deg_conn, np.mean(res.avg_deg_conn))
        rstats.largest_subgraph = np.append(rstats.largest_subgraph, np.mean(res.largest_subgraph))
        rstats.frequencymap = res.frequency
        rstats.runtime = np.append(rstats.runtime, res.runtime)

        n = (res.frequency.shape[0]-2)**2
        ji = ( ( np.sum(res.frequency[1:-1,1:-1] )**2 )/ ( n * np.sum( np.square( res.frequency[1:-1,1:-1] ) ) ) )        
        rstats.jain_index = np.append(rstats.jain_index, ji )  # IMP: Jain is used for calculation coverage fairness
        
        rstats.biconnected_gaint=np.append(rstats.biconnected_gaint, np.sum(res.is_biconnected_Gaint_Comp))

        # Time connected to any Base Station
        connected_any_bs = np.any(res.total_time_connected_to_BS, axis=1)
        total_time_connected_any_bs_per_node = np.sum(connected_any_bs[:, 1:], axis=1)
        avg_time_connected_any_bs = np.mean(total_time_connected_any_bs_per_node)
        
        rstats.percent_time_connected_to_BS.append(avg_time_connected_any_bs)
        
        # Nodes connected to any Base Station
        total_nodes_connected_any_bs = np.sum(connected_any_bs[:, 1:], axis=0)
        avg_nodes_connected_any_bs = np.mean(total_nodes_connected_any_bs)
        
        rstats.percent_nodes_connected_to_BS.append(avg_nodes_connected_any_bs)
        
        # Time connected to each Base Station
        total_time_per_bs_per_node = np.sum(res.total_time_connected_to_BS[:, :, 1:], axis=2)
        avg_time_node_connected_per_bs = np.mean(total_time_per_bs_per_node, axis=0)
        
        rstats.percent_time_connected_to_BSID.append(avg_time_node_connected_per_bs)
        
        # Nodes connected to each Base Station
        total_nodes_per_bs_per_t = np.sum(res.total_time_connected_to_BS[:,:,1:], axis=0)
        avg_nodes_per_t_connected_per_bs = np.mean(total_nodes_per_bs_per_t, axis=1)
        
        rstats.percent_nodes_connected_to_BSID.append(avg_nodes_per_t_connected_per_bs)
    
    rstats.percent_time_connected_to_BS = np.array(rstats.percent_time_connected_to_BS)
    rstats.percent_nodes_connected_to_BS = np.array(rstats.percent_nodes_connected_to_BS)
    rstats.percent_time_connected_to_BSID = np.array(rstats.percent_time_connected_to_BSID).T
    rstats.percent_nodes_connected_to_BSID = np.array(rstats.percent_nodes_connected_to_BSID).T

    rstats.biconnected_gaint /= simTime
    
    # Get time taken to cover 90% of map
    for s,r in enumerate(rstats.coverage):
        for t,v in enumerate(r):
            if v >= 90:                    # 90% of map  
                rstats.timeto90[s]=t
                break

    #Plot and save logged stats--- (***For not showing and saving figures use argument - noshow = False )---
    # save rstats as object
    save_object(rstats, save_path+'av{}-Qstat.pickle'.format(beta_type))
    with open(save_path + f"av{beta_type}-output.txt", 'w') as f:
        print_stats(rstats, f) 

    # For plotting results   
    plotting.plot_performance_plots(rstats,nAgents, save_path, noshow=True)  
    # plotting.plot_Connectivity_Histogram(globals.connectivity_histogram, save_path, noshow=True)

# ________________________________________________________________________         
                           
# ------------------------------------------------------------------------------------------------------------------------------------------- 
# main -end    

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)  # In case NumPy is used in computations

def save_NodeTrajectories(file_path, MOBILILTY_MODEL_TYPE, runNo, nAgents, log_positions, log_nextwaypoints, log_startTime, log_endTime, ):
    os.makedirs(file_path, exist_ok=True)

    for node in range(nAgents):
        A = log_positions.transpose(1, 0, 2)  # Swap (t, n, x) → (n, t, x)
        B = log_nextwaypoints.transpose(1, 0, 2)  # Swap (t, n, x) → (n, t, x)
        NODE_LOCATIONS_WAYPOINTS = np.concatenate((A[node], B[node]), axis=1)
        # print(NODE_LOCATIONS_WAYPOINTS.shape)

        # Corrected slicing using log_startTime, log_endTime and and division by 1000 to save in km.
        np.savetxt(file_path + f"{runNo}{MOBILILTY_MODEL_TYPE}_Node{node}.txt", (NODE_LOCATIONS_WAYPOINTS[log_startTime:log_endTime]) / 1e3, delimiter=",", fmt="%.5f")
        # Saved data is the following format: each row(time) = position_X, position_Y, nextwaypoint_X, nextwaypoint_Y
        

def print_stats(rstats, f = sys.stdout):
    with np.printoptions(threshold=np.inf):
        # print("coverage", rstats.coverage, file=f)
        print( "nnc= ",np.mean(rstats.no_connected_comp), "anc= ",np.mean(rstats.avg_deg_conn),\
            "gaint= ",np.mean(rstats.largest_subgraph), "'%' is_biconnected_gaint",  rstats.biconnected_gaint, file=f)
        print("jainsindex= ",np.mean(rstats.jain_index)) #, "  recent coverage= ",rstats.cellscoverage_per_100s, file=f)
        print("mean Time for 90% coverage (T90)= ",np.mean(rstats.timeto90), "  std T90=", np.std(rstats.timeto90), file=f)   
        print("avg time_connected_to_BS", np.mean(rstats.percent_time_connected_to_BS) , np.std(rstats.percent_time_connected_to_BS), file=f)
        print("avg nodes_connected_to_BS at any given time", np.mean(rstats.percent_nodes_connected_to_BS) , np.std(rstats.percent_nodes_connected_to_BS), file=f)

 # save rstats as object         
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    print(obj)
    return obj        


# other function and class definitions used for logging stats

        
class R_Stats(object):
    def __init__(self, num_episodes, simTime):
        self.coverage = np.empty((0, int(simTime/STATS_INTERVAL)+1 ));  # STATS_INTERVAL=100
        self.no_connected_comp = [] # NCC
        self.avg_deg_conn = []      # AND/ANC
        self.largest_subgraph = []
        self.biconnected_gaint=[]

        self.frequencymap = []    
        self.jain_index = []     # JAIN INDEX
        # self.cellscoverage_per_100s=np.empty((0, int(simTime/100) ))
        # self.fairness= np.empty((0, int(simTime/STATS_INTERVAL) ))
        self.timeto90=np.zeros(num_episodes)        # time taken for 90% of map coverage

        self.percent_time_connected_to_BS = []
        self.percent_nodes_connected_to_BS = []   
        self.percent_time_connected_to_BSID = []
        self.percent_nodes_connected_to_BSID = []
        
        self.runtime = []

#################################- MAIN-EXCUTION -##################################################
if __name__ == "__main__":
    # running_as = 'M'
    # if os.getenv("RUN_BY_SUBPROCESS"):
    #     running_as = 'SP'
    #     if os.getenv("SWEEP_NAME"):
    #         running_as = os.getenv("SWEEP_NAME")
    # #np.random.seed(0);
    tic = time.time();

    main()
    
    print("Elapsed time = ",time.time()-tic)
#    Z=input("hit enter to close")
    sys.exit(0)


