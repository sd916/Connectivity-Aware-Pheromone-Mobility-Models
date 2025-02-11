#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 8 2023
Modified on Feb 1 2025
@author: Shreyas Devaraju (devaraju@uci.edu)
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Modified for Connectivity-Oriented Pheromone based Movement simulation
# Shreyas Devaraju, Feb 8 2023
# Commented and cleaned on March 25 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import itertools
import matplotlib.pyplot as plt

import numpy as np
# from scipy.signal import convolve2d
from scipy.ndimage import convolve as convolveim
import time
import random
import networkx as nx
# import q_agent_V2 as ql
import globals # contains global variable
from numba import jit

from celluloid import Camera

arr = np.asarray

from line_profiler import profile

try:
    @profile
    def check_for_line_profiler():
        pass
except:
    def profile(f):
        return f



# For saving stats/results of each run
class UAVStats(object):
    def __init__(self, nAgents, nBaseStations, xgv, ygv, sim_time): #BSedit
        self.coverage = [0]
        self.cellscoverage_per_100s = []
        #self.coverage_rate=[]
        self.visited = np.zeros((len(xgv)+2, len(ygv)+2))  # __grid;
        self.frequency = np.zeros((len(xgv)+2, len(ygv)+2))  # __grid;
        self.fairness = []
        #self.temp_100s = __grid;
        self.largest_subgraph = []
        self.no_connected_comp = []
        self.freq_subgraph_sizes = np.zeros(nAgents-1)
        self.amt_of_k_connectivity_of_largest_component = 0
        self.avg_deg_conn = []
        self.is_biconnected_Gaint_Comp = []
        self.cell_visted_times = np.zeros((len(xgv)+2, len(ygv)+2, 4000))

        self.total_time_connected_to_BS = np.zeros((nAgents, nBaseStations, sim_time+1)) # last node id is BS itself #BSedit
        # pass
        
        self.runtime = []


    #----------------------------------------------------------------------------------
        self.percentage_coverage_in_episode=0.0
        #Log UAV position and next-waypoints
        self.UAV_positions = np.zeros((sim_time+1, nAgents, 2))  
        self.UAV_nextwaypoints = np.zeros((sim_time+1, nAgents, 2))  

        pass


class uav_swarm(object):
    """Multi-agent UAV swarm simulation using pheromone tracking"""

    def __init__(self, env, **kwargs):
        self.env = env
        
        self.stats = None

        self.evaporation_rate  = kwargs.get('evaporation_rate',    .1)
        self.diffusion_rate    = kwargs.get('diffusion_rate',      .1)

        self.time_step    = kwargs.get('time_step',      1)
        self.stats_interval = kwargs.get('stats_interval',  100)  # stats log interval
        self.hello_period = kwargs.get('hello_period',   2)      # hello message interval
        #self.sim_time     = kwargs.get('sim_time',     400)  # made sim parameter
        self.nAgents      = kwargs.get('nAgents',       20)
        self.nBaseStations= kwargs.get('nBaseStations', 0)  # For CAP we have zero base station node in map; BS-CAP we need atleast 1 BS
        self.nHistory     = kwargs.get('nHistory',      40)
        self.use_pheromone= kwargs.get('use_pheromone',True)  # use pheromone flag
        self.use_connect  = kwargs.get('use_connect', True)   # use connectivity constraint flag
        self.min_degree   = kwargs.get('min_degree',     2)
        self.map_size     = kwargs.get('map_size',    6000)  # map size
        self.map_resolution=kwargs.get('map_resolution',100) # cell resolution
        self.transmission_range=kwargs.get('transmission_range',1000) # Tx range
        self.fwd_scheme   = kwargs.get('fwd_scheme',     5)  # 5 forward waypoint selection
        self.hop_dist     = kwargs.get('hop_dist',       2)  # next waypoint cell 2 cells away; Also affects get_neighbour_pheromone_weight funtion
        self.collision_avoidance = kwargs.get('collision_avoidance', True)
        self.turn_buffer  = kwargs.get('turn_buffer',  40)   # buffer to start turn at boundary
        self.collision_buffer = kwargs.get('collision_buffer', 40) # buffer to avoid UAV collision
        self.uav_airspace = self.collision_buffer   #
        self.waypoint_radius = kwargs.get('waypoint_radius', 50) # tollerance distance around next waypoints
        self.uav_speed = kwargs.get('uav_speed', 20) # UAv SPEED - 20 , 40 m/s
        self.skip_stats = kwargs.get('skip_stats', 0)
        self.MOBILILTY_MODEL_TYPE = kwargs.get('MOBILILTY_MODEL_TYPE', 'CAP')
        
        self.camera = None
        self.fig = None

        self.xgv, self.ygv = [],[]  # resolution / locations for grids
        self.xvv, self.yvv = [],[]  # alternate grid? (unused)

        self.colors = np.array([[0.859,0.251,0.251], [0.969,0.408,0.408], [1.0,0.588,0.588], # for plotting (not used)
                                [0.737,0.122,0.122], [0.585,0.035,0.035], [1.0,0.0,0.0],
                                [0.988,0.102,0.102], [1.0,0.266,0.266], [0.784,0.0,0.0]])

        self.init_posX= kwargs.get('init_posX',    np.array( (self.map_size/2) + (-1+2*np.random.rand(self.nAgents+self.nBaseStations))*200 ) )
        self.init_posY= kwargs.get('init_posY',    np.array( 300+ (-1+2*np.random.rand(self.nAgents+self.nBaseStations))*200 ) )

        assert self.nBaseStations <= 4 # Supports 4 BS for now

        self.bs_iDs = [self.nAgents+i for i in range(self.nBaseStations)]
        self.bs_positions = []
        if self.nBaseStations > 0:
            self.bs_positions.append([self.map_size/2, 0.0])
        if self.nBaseStations > 1:
            self.bs_positions.append([self.map_size/2, self.map_size])
        if self.nBaseStations > 2:
            self.bs_positions.append([0.0, self.map_size/2])
        if self.nBaseStations > 3:
            self.bs_positions.append([self.map_size, self.map_size/2])

        for i, bsID in enumerate(self.bs_iDs):
            self.init_posX[bsID] = self.bs_positions[i][0]   #BSedit BS Location
            self.init_posY[bsID] = self.bs_positions[i][1]   #BSedit

        self.init_map()
        self.init_agents()

        self.alpha_type = kwargs.get('alpha_type', 0)
        self.power = kwargs.get('power', 1)
        

    def init_map(self):
        """Re-initialize grids and pheromone maps"""
        self.xgv = np.linspace( 0,self.map_size, int(self.map_size/self.map_resolution) )  # grid size 1 unit
        self.ygv = np.linspace( 0,self.map_size, int(self.map_size/self.map_resolution) )  # grid size 1 unit
        # ATI: removed unused code to use a different grid size for visitation counts

        self.pheromone = self.__grid();   # pheromone map

        self.node_pheromone_map = np.tile(self.__grid()[:,:,np.newaxis],(1,1,self.nAgents+self.nBaseStations))
        self.node_pheromone_Attract = np.tile(self.__grid()[:,:,np.newaxis],(1,1,self.nAgents+self.nBaseStations))
        self.node_pheromone_Repel = np.tile(self.__grid()[:,:,np.newaxis],(1,1,self.nAgents+self.nBaseStations))
        self.node_pheromone_Tracking = np.tile(self.__grid()[:,:,np.newaxis],(1,1,self.nAgents+self.nBaseStations))


        #################################################################################

    def init_agents(self):
        """Initialize the state of the controllers & robots"""
        self.Acontroller = np.zeros( (14,self.nAgents+self.nBaseStations) )                      # controller & robot states
        self.Arobot = np.zeros( (5,self.nAgents+self.nBaseStations) )

        self.Arobot_prev_cell = np.zeros( (2,self.nAgents+self.nBaseStations) ).astype(int)
        self.Arobot_history = np.zeros( (self.nHistory,3,self.nAgents+self.nBaseStations) ) + np.nan  # save position history for drawing

        self.prev_state =  np.zeros((self.nAgents+self.nBaseStations, 2*self.fwd_scheme), dtype=np.ndarray) #(NOT USED in Heuristic Model)
        self.prev_action =  np.zeros((self.nAgents+self.nBaseStations), dtype=int)

        # Create the agents and initialize
        for uavID in range(self.nAgents):
            # Start positions
            self.Acontroller[0,uavID] = self.init_posX[uavID]
            self.Acontroller[1,uavID] = self.init_posY[uavID]
            self.Acontroller[2,uavID] = 0.0 #10 * (rand); (not used for now)          # z position
            self.Acontroller[3,uavID] = 0 # heading (theta)   /360 * rand;
            self.Acontroller[4,uavID] = 0 # not used (phi)
            # Commands
            self.Acontroller[5,uavID] = 0 # velocity
            self.Acontroller[6,uavID] = 0 # mu
            # Memory
            self.Acontroller[7,uavID] = uavID   # id
            self.Acontroller[8,uavID] = 1    # state
            self.Acontroller[9,uavID] = 0    # neighbor.id
            self.Acontroller[10,uavID] = np.inf # neighbor.distance
            # Waypoint related information
            self.Acontroller[11,uavID] = 0.0   # target x postion (waypoint)
            self.Acontroller[12,uavID] = 0.0   # target y postion (waypoint)
            self.Acontroller[13,uavID] = 0   # steps (waypoint)

            # Physical Robot state used in move()
            self.Arobot[0:3,uavID] = np.copy( self.Acontroller[0:3,uavID]) # position xyz
            self.Arobot[3,uavID] = 0                          # heading theta
            self.Arobot[4,uavID] = 0                          # phi

            self.Arobot_prev_cell[:,uavID]=[-1,-1]   # previous cell

        # Create the base stations and initialize
        for i, bsID in enumerate(self.bs_iDs):
            # Start positions
            self.Acontroller[0,bsID] = self.init_posX[bsID]
            self.Acontroller[1,bsID] = self.init_posY[bsID]
            self.Acontroller[2,bsID] = 0.0 #10 * (rand); (not used for now)          # z position
            self.Acontroller[3,bsID] = 0 # heading (theta)   /360 * rand;
            self.Acontroller[4,bsID] = 0 # not used (phi)
            # Commands
            self.Acontroller[5,bsID] = 0 # velocity
            self.Acontroller[6,bsID] = 0 # mu
            # Memory
            self.Acontroller[7,bsID] = bsID   # id
            self.Acontroller[8,bsID] = 1    # state
            self.Acontroller[9,bsID] = 0    # neighbor.id
            self.Acontroller[10,bsID] = np.inf # neighbor.distance
            # Waypoint related information
            self.Acontroller[11,bsID] = 0.0   # target x postion (waypoint)
            self.Acontroller[12,bsID] = 0.0   # target y postion (waypoint)
            self.Acontroller[13,bsID] = 0   # steps (waypoint)

            self.Acontroller[11,bsID] = self.bs_positions[i][0]   # target x postion (waypoint) #BSedit
            self.Acontroller[12,bsID] = self.bs_positions[i][1]   # target y postion (waypoint) #BSedit

            # Physical Robot state used in move()
            self.Arobot[0:3,bsID] = np.copy( self.Acontroller[0:3,bsID]) # position xyz
            self.Arobot[3,bsID] = 0                          # heading theta
            self.Arobot[4,bsID] = 0                          # phi

            self.Arobot_prev_cell[:,bsID]=[-1,-1]   # previous cell

        # print(f"Base stations = {self.nBaseStations}")
        ## Initialise communications
        self.channel = self.initChannel(self.nAgents + self.nBaseStations)
        # print(f"{self.channel.shape = }")



    # helper for creating properly-sized grids
    def __grid(self): return np.zeros((len(self.xgv)+2,len(self.ygv)+2))

    # UAV bounds:                      min speed   max speed  min turn   max turn
    def __boundv(self,dt): return arr( [  self.uav_speed*dt ,     self.uav_speed*dt ,   -3*dt  ,   3*dt   ] )  #SHREY: 20 , 40 m/s
    # def __initUAVfigure(self): plt.figure(self.fig); ax=plt.subplot(111,projection='3d'); plt.grid(True); return ax
    def __initUAVfigure(self): self.fig; ax=plt.subplot(111,projection='3d'); plt.grid(True); return ax


    @profile
    def sim_start_3d_simpy(self, sim_time, drawUAVs=True, drawMap=False, stats=None, plotInterval=10):
        """Run simulation of UAVs with pheromone-based directions"""

        starttime = time.time()
        # initialize statistics object if not passed in
        if stats is None:
            self.stats = UAVStats(self.nAgents, self.nBaseStations, self.xgv, self.xgv, sim_time) #BSedit

        # initiallize evp and diff rates
        evaporation_rate = self.evaporation_rate
        diffusion_rate = self.diffusion_rate

        ### coverage metrics
        temp_100s = np.zeros((len(self.xgv)+2, len(self.ygv)+2))#self.__grid(); #SD

        # delays = myDelay() # (not used)

        # ------------------------------------------------------------------------------------
        # Plotting flags -ON/OFF
        drawplots = drawUAVs
        drawAirspace = False           # SHREY draw circle range and repullision radius
        drawUAVflight = drawUAVs
        drawUAVconnectivity = drawUAVs
        drawPheromonemap = drawMap  # true or false for image of pheromone map
        plot_interval = plotInterval
        # ax,axx = None,None     # pre-plotting, no axes exist yet
        ax, axx = None, (70, 5)    # initial elevation & azimuth
        if (drawplots):
            plt.figure(1)
        # ------------------------------------------------------------------------------------

        ## Initialise Time
        t = 0
        dt = self.time_step
         # UAV positions at t =0
        self.stats.UAV_positions[0,:,:]= self.Arobot[0:2, :].T
        ## Main simulation loop
        while True:
            t = int(self.env.now)
            tidx = t
            if t%100==0:print(t)
            node_pheromone_map_Tminus1 = np.copy(self.node_pheromone_map)
            for uavID in range(self.nAgents):  #BSedit
            
                #   print(f'{t} : {uavID} [{str(self.Acontroller[:,uavID])}]');
                ## Get simulation values for the node
                controller = self.Acontroller[:,uavID]
                robot = self.Arobot[:,uavID]
                # Prev_state, action of each UAV for Q-learning #(NOT USED in Heuristic Model)
                prev_state = self.prev_state[uavID]
                prev_action = self.prev_action[uavID]

                ## Take measurement
                ## Controller
                msgs = self.simReceive(self.channel);            # Receive messages from other agents
                if(t > 2):
                    msgs[:,uavID] = [ controller[0],controller[1],controller[2],controller[3],controller[4],controller[5],controller[6],uavID,controller[11],controller[12] ]
                    for i in range(len(self.bs_iDs)):
                        msgs[:,self.bs_iDs[i]] = [ self.bs_positions[i][0], self.bs_positions[i][1],0,0,0,0,0,self.bs_iDs[i], self.bs_positions[i][0], self.bs_positions[i][1] ]  #BSedit

                # Decide where to move next
                prev_state, prev_action = self.decide_3d(uavID, controller, msgs, dt, prev_state, prev_action)

                if prev_state is not None: # if prev_state not None, i.e. its returned a new state to be stored to memory #(NOT USED in Heuristic Model)
                    # Store each uav's previous state
                    self.prev_state[uavID] = prev_state
                    self.prev_action[uavID] = prev_action

                # Update position estimates
                k = controller[5] * controller[6]
                controller[3] += (k + 2*k + 2*k +k)*dt/6
                controller[3] = controller[3] % 360

                ## Physical Robot
                self.Arobot_history[tidx%self.nHistory,:,uavID] = robot[0:3];    # store in history queue

                #% Move the robot
                robot = self.move(robot, controller[5], controller[6], dt)

                #----------- PHEROMONE Depoit and distribution----------------------------------------------------------------------------
                pher_cell = tuple(np.ceil( robot[0:2] / self.map_resolution ).astype(int))
                #Shrey:
                if (self.use_pheromone):
                        #(curr_cellP ~= robot.prev_cell) condition allows for only one deposit per visit by a uav.
                        if (np.all(arr(pher_cell)>=1) and np.all(arr(pher_cell)<=(self.pheromone.shape[1] -2))): #62-2=60=>range between [1,60]
                            if np.any(pher_cell != self.Arobot_prev_cell[:,uavID]):
                                self.Arobot_prev_cell[:,uavID] = pher_cell

        #12                        self.node_pheromone_Repel[pher_cell+(uavID,)] = 1;   #node_pheromone_Repel(pher_x,pher_y,uavID) + 10 ; %shrey: %NEW increament pheromone by 1

                                # Deposit pheromone in cell and increment cell visitation counters for output calculation
                                i = pher_cell[0] #cell index
                                j = pher_cell[1]

                                self.node_pheromone_Repel[i, j, (np.ones(9)*uavID).astype(int)] = 1.0  # Deposit Repel pheromone
                                
                                if t >= self.skip_stats:
                                    self.stats.visited[ i, j ] = 1  # cell scanned atleat once

                                    self.stats.cell_visted_times[i,j,self.stats.frequency[ i, j ].astype(int)] = t # cell last visited time
                                    self.stats.frequency[ i, j ] += 1 # cell visit freq
                                temp_100s[ i, j ] = 1

                            else:
                                self.node_pheromone_Repel[pher_cell+(uavID,)] = 1 #node_pheromone_Repel(pher_x,pher_y,uavID) /((1-evaporation_rate) * (1-diffusion_rate));

                else:
                    if (np.all(pher_cell>=0) and np.all(pher_cell<len(self.stats.visited))):   # SHREY: BUG-FIX >= , <
                        raise NameError('one count per visit by a uav')   # ATI: what is this doing?


                ## Controller
                controller[0:2] = self.gps(robot)[0:2] # Retrieve noisy location from GPS;  Shrey: removed gps noise

                # Send location to other agents
                msg = [controller[0],controller[1],controller[2],controller[3],controller[4],controller[5],controller[6],uavID,controller[11],controller[12] ];
                #self.simTransmit(self.channel, uavID, msg);
                if(t % self.hello_period == 0):
                    self.simTransmit(self.channel, uavID, msg)
                
                    for i in range(self.nBaseStations):            
                        self.simTransmit(self.channel, self.bs_iDs[i], [ self.bs_positions[i][0], self.bs_positions[i][1],0,0,0,0,0,self.bs_iDs[i], self.bs_positions[i][0], self.bs_positions[i][1] ])  #BSedit
            
                ## Store values
                self.Acontroller[:,uavID] = controller
                self.Arobot[:,uavID] = robot


            #------------PHEROMONE COMPUTATION $ ITS PERIODIC DISTRIBUTION--------------------------------------
            if (self.use_pheromone):
                #Check connectivity and merge pheromone map of connected UAV neighbors, and this update is done every hello_period=4s,2s (virtual hello packet)
                if ( (t % self.hello_period) == 0 ):
                    connMat = self.connectivity(self.Arobot[0:2,:],self.transmission_range);     #check connected neighbors
                    for uavID in range(self.nAgents+self.nBaseStations):
                        conn_neighbors = np.flatnonzero(connMat[:,uavID]);    # get neighbors of uavID
                        self.merge_pheromone_map(uavID, conn_neighbors)       # function merges pheromone map of connected UAV neighbors

                #  Difussion and evaporation for every node_pheromone_Repel,
                for uavID in range(self.nAgents+self.nBaseStations):
                    h=np.ones((3,3))
                    h[1,1]=0
                    # ATI faster
                    self.node_pheromone_Repel[:,:,uavID]= ( (1-evaporation_rate) * ( (1- diffusion_rate)*self.node_pheromone_Repel[:,:,uavID] + (diffusion_rate/8)*convolveim(node_pheromone_map_Tminus1[:,:,uavID], h, mode='constant') ) ); #%shrey  % NEW(1-evaporation_rate)
                    i =self.Arobot_prev_cell[0,uavID];j=self.Arobot_prev_cell[1,uavID]
                    self.node_pheromone_Repel[  i, j, (np.ones(9)*uavID).astype(int)]  = 1.0
                    self.node_pheromone_map[:,:,uavID] = self.node_pheromone_Repel[:,:,uavID]

            # Sanity Check
            if np.any(self.node_pheromone_Repel >1): raise NameError('pheromone value in cell > 1')

            # Update messages
            #self.simChannel(self.channel);
            if(t % self.hello_period == 0): self.simChannel(self.channel) # # Update messages every hello period

            # calculate Connectivity Matrix
            connMat = self.connectivity(self.Arobot[0:2,:self.nAgents],self.transmission_range)
            connMatwBS = self.connectivity(self.Arobot[0:2,:],self.transmission_range, connect_base_stations=True)
            # # print(t, connMat)

            #Ploting UAV figures---------------------------------
            self.plot_UAV_figures(t,tidx, ax, axx, drawplots, plot_interval, drawPheromonemap, drawUAVflight, drawAirspace, drawUAVconnectivity, connMatwBS)  #BSedit

            #log stats and metrics---------------------------------
            # stats = self.evalutaion_metric_calc( t , connMat, stats)
            temp_100s = self.evalutaion_metric_calc( t , connMat, connMatwBS, temp_100s)
            
            #log UAVPositions
            self.stats.UAV_positions[t,:,:] = self.Arobot[0:2, :].T  #SHREY* Update node locations every time step 
            self.stats.UAV_nextwaypoints[t,:,:] = self.Acontroller[11:13, :].T   #SHREY* nxtwaypoint = (controller[11], controller[12]); Update node next waypoint locations
            

            if t == sim_time - 1:
                self.stats.runtime.append(time.time() - starttime)
            
            yield self.env.timeout(dt)
    
 
    # function logs performance stats---------------------------------
    @profile
    def evalutaion_metric_calc(self, t , connMat, connMatwBS, temp_100s):
        if (t % self.stats_interval== 0):                                        # Save value every 'x' sec
            #coverage percentage
            self.stats.percentage_coverage_in_episode= (self.stats.visited[1:-1,1:-1].sum()/(self.stats.visited[1:-1,1:-1].size))*100
            self.stats.coverage.append(self.stats.percentage_coverage_in_episode);
            #fairness
            self.stats.fairness.append( np.std(self.stats.frequency[1:-1,1:-1]) )
        if t%100==0:
            #recent coverage every 100sec
            self.stats.cellscoverage_per_100s.append(temp_100s[1:-1,1:-1].sum());
            temp_100s = np.zeros((len(self.xgv)+2, len(self.ygv)+2))#self.__grid(); #SD

        # Save Connectivity output values every second
        nbins,bins,binsizes, is_biconnected_Gaint = self.myconncomp(connMat);
        self.stats.no_connected_comp.append(nbins);                # NCC
        self.stats.largest_subgraph.append(max(binsizes));

        subgraphhist,_ = np.histogram(binsizes, np.arange(self.nAgents)+.5)    # get histogram of subgraph sizes
        self.stats.freq_subgraph_sizes += subgraphhist;         # and add to get frequency statistics

        deg = connMat.sum(1);                                    # get avg degree of connectivity of network
        self.stats.avg_deg_conn.append( np.mean(deg) );  # ANC

        self.stats.is_biconnected_Gaint_Comp.append(is_biconnected_Gaint)

        closure = warshall(connMatwBS) # get transitive closure of network
        for node in range(self.nAgents):
            for i in range(len(self.bs_iDs)):
                if closure[node, self.bs_iDs[i]] == 1:
                    self.stats.total_time_connected_to_BS[node, i, t] = True

        return temp_100s

    ## Adjacency matrix of communcation connectivity
    def connectivity(self, locations, t_range, connect_base_stations=False):
        D,nAgents = locations.shape;
        conn = ((locations.reshape((D,nAgents,1)) - locations.reshape((D,1,nAgents)))**2).sum(0) < (t_range**2);
        conn[ range(nAgents),range(nAgents) ] = 0; # remove self-edges
        if connect_base_stations:
            for bs1, bs2 in itertools.combinations(self.bs_iDs, 2):
                conn[bs1, bs2] = 1
                conn[bs2, bs1] = 1
        return conn

    def estimate_node_connectivity_at_nextwaypoints(self, node_id, nxt_waypoint, msgs_neighbor_nxt_waypoints, dt, t_range):
        """ Gives the node's distance weighted connectivity wrt to the node's next waypoint and neighbor nodes next waypoint.

        Args:
            node_id: node id
            nxt_waypoint:  node's next waypoint
            msgs_neighbor_nxt_waypoints: neighbour nodes next waypoints sent through hello messages
            dt: time step (NOT USED)
            t_range: transmission range ;default value-1000m

        Returns:
            dst-weighted-connectivity value :(float).

        """
        nbrs = np.sqrt( ( ( msgs_neighbor_nxt_waypoints[8:10,:] - arr(nxt_waypoint[0:2]).reshape(2,1) )**2 ).sum(0) ).astype(float)
        # print(nbrs)
        d1 = nbrs <= (0.6 * t_range)
        d2 = ((nbrs >  (0.6 * t_range)) & (nbrs <= t_range))
        d3 = nbrs > t_range
        nbrs[d1] = 1.0
        nbrs[d2] = 2.5*(1. - nbrs[d2]/t_range)  #nbrs[d2] * (-1.0/400.0) + 2.5 #For TX_RANGE=1000m
        nbrs[d3] = 0.0
        # print(nbrs)
        try:
            nbrs[int(node_id)] = 0.0;  # not our own neighbor
        except:
            if len(nbrs)==0:return 3.0
        return nbrs.sum();


    def node_distance_weighted_connectivity_at_current_position(self, node_id, curr_position, t_range):
        """ Gives the node's distance weighted connectivity wrt to the node's curent position and neighbor nodes current taken

        Args:
            node_id: node id
            curr_position:  node's current position coordinates[x,y]
            t_range: transmission range ;default value-1000m.

        Returns:
            dst-weighted-connectivity value :(float).

        """
        neighbor_nodes_current_positions = self.Arobot[0:2,:] # list of neighbour nodes current positions
        nbrs = np.sqrt( ( ( neighbor_nodes_current_positions - arr(curr_position[0:2]).reshape(2,1) )**2 ).sum(0) ).astype(float)
        # print(nbrs)
        d1 = nbrs <= (0.6 * t_range)
        d2 = ((nbrs >  (0.6 * t_range)) & (nbrs <= t_range))
        d3 = nbrs > t_range
        nbrs[d1] = 1.0
        nbrs[d2] = 2.5*(1. - nbrs[d2]/t_range)  #nbrs[d2] * (-1.0/400.0) + 2.5 #For TX_RANGE=1000m
        nbrs[d3] = 0.0
        # print(nbrs)
        nbrs[int(node_id)] = 0.0;  # not our own neighbor
        return nbrs.sum()

    def nodes_distance_from_me_Arobot_locations(self, node_id, curr_position):
        """ Gives all the other node's distance wrt to the node's curent position and hello msgs

        Args:
            node_id: node id
            curr_position:  node's current position coordinates[x,y]

        Returns:
            other node's distance wrt to the node's curent position :(list of float values).

        """
    #      neighbor_nodes_current_positions = hello_msgs[0:2,:] # list of neighbour nodes cuurrent positions
        all_nodes_current_positions = self.Arobot[0:2,:] # list of all nodes current positionss
        nbrs = np.sqrt( ( ( all_nodes_current_positions - arr(curr_position[0:2]).reshape(2,1) )**2 ).sum(0) ).astype(float)

    #   nbrs[int(node_id)] = 0.0;  # not our own neighbor
        return nbrs

    def myconncomp(self, adj):
        import networkx as nx
        G = nx.from_numpy_array(adj); # Return a graph from numpy matrix
        CCs = [c for c in nx.connected_components(G)];  # convert generator to list if required
        nbins = len(CCs);
        bins = np.zeros(adj.shape[0]);
        binsizes = np.zeros(nbins);
        for i,c in enumerate(CCs): bins[list(c)]=i; binsizes[i]=len(c);

        largest_cc = max(nx.connected_components(G), key=len)
        is_biconnected_Gaint_Comp = nx.is_biconnected(G.subgraph(largest_cc))
        return nbins,bins,binsizes, is_biconnected_Gaint_Comp


    ## Communications
    def initChannel(self,nAgents):
        return np.zeros((10,nAgents,2)) + np.nan;

    def simReceive(self,channel):
        keep = ~np.isnan(channel[0,:,0]);
        return channel[:,keep,0];

    def simTransmit(self,channel, uavID, txMsgs):
        channel[:,uavID,1] = txMsgs;

    def simChannel(self,channel):
        channel[:,:,0] = channel[:,:,1];
        channel[:,:,1] = np.nan;

    def gps(self, robot, noise=0):
        """A GPS is accurate to between +/- 3mon a good day"""
        return robot + (np.random.random(robot.shape)-.5)*noise


    @profile
    def decide_3d(self, uavID, controller, msgs, dt, prev_state, prev_action):
        """  """
        b = self.__boundv(dt);

        #% Agent finite state machine
        if (controller[8]==1):
            #case 1
            #% Pick a random target way point

            curr_cell= np.ceil(controller[0:2]/self.map_resolution).astype(int)

            if self.use_pheromone:  # ATI weird, no difference
                w= self.get_neighbor_pheromone_weight(controller[7], curr_cell);
            else:
                w= self.get_neighbor_pheromone_weight(controller[7],curr_cell);

            R = np.random.choice( np.flatnonzero(w==min(w)) );

            controller[11],controller[12] = self.return_next_dst_point_based_on_direction(curr_cell, R, self.hop_dist); # return x,y coordinates of UAVs next destination point

            controller[13] = 0;
            controller[8] = 2; # UAV controller state to -Fly to target waypoint

        elif (controller[8]==2): #% Fly to target waypoint
            #case 2
            #% Travel in a straight line at top speed
            controller[6] = 0;
            controller[5] = b[1]; #%.maxv(dt);

            #% Turns required?
            if controller[13] > 0:
                controller[13] -= 1;
            else:
                [controller, steps] = self.face(controller, controller[11], controller[12], dt); #%SHREY: turn on & face towards the target(x,y)
                controller[13] = steps - 1;

            #% Too close to the boundary?
            #if (np.any(controller[0:2]>self.map_size) or np.any(controller[0:2] < 0)):
                #print('[Agent {}] I am OUTSIDE the boundary!'.format(controller[7]));


            #% Where will we be next timestep (after dt =1sec)?
            x, y, theta = rk4(controller[0], controller[1], controller[3], controller[5], controller[6], dt);
            nextpos = arr([x,y,theta]);


            #% SHREY : random direction selection untill it turns away from boundary
            if (np.any(nextpos[0:2] > self.map_size-self.turn_buffer) or np.any(nextpos[0:2]<0+self.turn_buffer)): #SHREY: FIX if (np.any(nextpos[0:2] > self.map_size-self.turn_buffer) or np.any(nextpos[0:2]<0+self.turn_buffer)):
                controller[8] = 1;

            if self.collision_avoidance:
                #% Check too close to another agent?
                [controller, stop] = self.airspace(controller, nextpos, msgs, dt, self.uav_airspace);
                if stop: return None,None

            #% When we have arrived just before the next waypoint ***, then do the following:
            #% (Note:Use a threshold to stop the UAV spinning around the target waypoint.)

            if( np.sqrt(((nextpos[0] - controller[11])**2) + ((nextpos[1] - controller[12])**2)) < self.waypoint_radius ):
                # UAVs current cell
                curr_cell= np.ceil(controller[0:2]/self.map_resolution).astype(int)

                #% Get the pheromone weight in each direction
                if self.use_pheromone:
                    wgt_avg_pheromone_values = self.get_neighbor_pheromone_weight(controller[7],curr_cell);
                else:
                    wgt_avg_pheromone_values = np.zeros((1,8));  # Random direction selection;

                #% Choose the direction based on pheromone or pheromone+connectivity condition.
                if self.use_connect:
                    curr_heading_R =  int((controller[3] + 22.5)/45) % 8;  # UAV current heading

                    if (self.fwd_scheme == 3):   # find pheromone from cells in possible UAV heading directions (for 3 forward headings)
                        waypoint_directions = arr([curr_heading_R-1, curr_heading_R, curr_heading_R+1]) % 8;
                        wgt_avg_pheromone_values_at_nxt_waypoints = wgt_avg_pheromone_values[ waypoint_directions ]; # pheromone value at next waypoint cells

                    elif (self.fwd_scheme == 5): # or: (for 5 forward headings)
                        waypoint_directions = arr([curr_heading_R-2, curr_heading_R-1, curr_heading_R, curr_heading_R+1, curr_heading_R+2]) % 8
                        wgt_avg_pheromone_values_at_nxt_waypoints = wgt_avg_pheromone_values[ waypoint_directions ];

                    else:
                        raise ValueError('code suports only fwd=3 or 5');

                    # SELECT MOBILILTY_MODEL_TYPE = 'CAP' or 'BSCAP' mobility model
                    if self.MOBILILTY_MODEL_TYPE == 'CAP':
                        connectivity_at_nxt_waypoints=[]  # connectivity at possible next waypoints
                        for direction_R in waypoint_directions:
                            nxt_waypoint = [None, None]
                            nxt_waypoint[0], nxt_waypoint[1] = self.return_next_dst_point_based_on_direction(
                                curr_cell, direction_R, self.hop_dist)             # get possible nextwaypoint locations (X,Y) based on current heading direction
                            dist_wt_connectivity = self.estimate_node_connectivity_at_nextwaypoints(
                                controller[7], nxt_waypoint, msgs, dt, self.transmission_range)         # get connectivity at nextwaypoint flag
                            connectivity_at_nxt_waypoints.append(
                                dist_wt_connectivity)
                                
                        # getting the current state_________________________________________________________________________
                        state_weighted_avg_pheromone = wgt_avg_pheromone_values_at_nxt_waypoints
                        # Weighted degree of connectivity at next waypoints at t+dt
                        state_connectivity = connectivity_at_nxt_waypoints

                        state_conn = np.copy(arr(state_connectivity).astype(int))
                        state_conn[np.where(state_conn > 10)] = 10
                        for i in state_conn:
                            globals.connectivity_histogram[i] += 1

                        # current_state = state_weighted_avg_pheromone
                        current_state = np.concatenate((arr(state_weighted_avg_pheromone), arr(state_connectivity)))  # state-s

                        Rdeg = np.copy(connectivity_at_nxt_waypoints)   # connectivity at nxtwayoints
                        pher_ww = np.copy(wgt_avg_pheromone_values_at_nxt_waypoints)   # pheroomone at nxtwayoints
                        alpha = []
                        # alpha = list(map(self.alpha_func, Rdeg))
                        alpha = np.array([self.alpha_func_CAP(r) for r in Rdeg])
                        alpha = np.array(alpha)
                        pher_ww = np.clip(pher_ww, 0, 1)
                        Pi = (alpha*(1-pher_ww))
                    
                    elif self.MOBILILTY_MODEL_TYPE == 'BSCAP':
                        connectivity_at_nxt_waypoints=[]  # connectivity at possible next waypoints
                        bs_connected_at_nxt_waypoints=[]  #BSedit - BS connectivity at possible next waypoints
                        dist_to_bs=[]  #BSedit - Distance to BS from possible next waypoints


                        uav_pos_now= np.copy(self.Arobot[0:2,:]) #current UAV positions
                        connMat_now = self.connectivity(uav_pos_now,self.transmission_range)  # Connectivity Matrix
                        G_now = nx.from_numpy_array(connMat_now)                             #  Network Graph from Connectivity Matrix (current UAV positions)

                        connMat_next = self.connectivity(msgs[8:10,:],self.transmission_range)  # UAV positions based on nextwaypoint from hello messages
                        # connMat_next = np.delete(connMat_next, uavID, axis=0)
                        # connMat_next = np.delete(connMat_next, uavID, axis=1)
                        G_next = nx.from_numpy_array(connMat_next)
                        G_next.remove_node(uavID)                                          #  Network Graph from Connectivity Matrix (UAV positions based on next waypoints)

                        for direction_R in waypoint_directions:
                            nxt_waypoint=[None,None];
                            nxt_waypoint[0],nxt_waypoint[1] = self.return_next_dst_point_based_on_direction( curr_cell, direction_R, self.hop_dist); # next-waypoint
                            dist_wt_connectivity = self.estimate_node_connectivity_at_nextwaypoints(controller[7] , nxt_waypoint, msgs ,dt, self.transmission_range );  # Estimated dst-wt-degree-of-connectivity at next wayponit
                            connectivity_at_nxt_waypoints.append(dist_wt_connectivity)
                            connected_bsID, is_connected = self.base_station_connected( uavID, msgs, nxt_waypoint )
                            bs_connected_at_nxt_waypoints.append(is_connected)  #BSedit - BS connectivity

                            if connected_bsID >= 0:
                                dbs=np.sqrt( np.sum( np.square(arr(self.bs_positions[connected_bsID])-arr(nxt_waypoint[0:2])) )).astype(float)  #BSedit - distance to BS fron next-waypoint
                                dist_to_bs.append(dbs)
                            else:
                                dbs = []
                                for i in range(self.nBaseStations):
                                    dbs.append( np.sqrt( np.sum( np.square(arr(self.bs_positions[i])-arr(nxt_waypoint[0:2])) )).astype(float) )
                                dist_to_bs.append(min(dbs))

                        bs_connected_at_nxt_waypoints=np.array(bs_connected_at_nxt_waypoints)  #BSedit

                        ## getting the current state___(State is NOT USED in Heuristic Model) ______________________________________________________________________
                        state_weighted_avg_pheromone = wgt_avg_pheromone_values_at_nxt_waypoints
                        state_connectivity = connectivity_at_nxt_waypoints #  Weighted degree of connectivity at next waypoints at t+dt

                        state_conn = np.copy(arr(state_connectivity).astype(int))
                        state_conn[ np.where(state_conn > 10) ] =10
                        for i in state_conn:
                            globals.connectivity_histogram[i] += 1

                        #current_state = state_weighted_avg_pheromone
                        current_state = np.concatenate((arr(state_weighted_avg_pheromone), arr(state_connectivity)) ) # state-s (is NOT USED in Heuristic Model)
                        ##_______________________________________________________________________________________________________________________________________

                        Rdeg = np.copy(connectivity_at_nxt_waypoints);
                        pher_ww = np.copy(wgt_avg_pheromone_values_at_nxt_waypoints)
                        alpha = [];
                        alpha = list( map(self.alpha_func_BSCAP, Rdeg) )
                        alpha = np.array(alpha);

                        pher_ww = np.clip(pher_ww, 0, 1)  # restrict pheromone values between 0 and 1
                        if np.any(pher_ww >1): raise ValueError("pher_ww > 1") # pher_ww=np.clip(pher_ww, 0, 1)#raise ValueError("pher_ww > 1")

                        Pi = ( alpha*(1-pher_ww))
                        # Pi=Pi/np.sum(Pi)

                        # CHECK BS connectivity True or False  #BSedit
                        if np.all(bs_connected_at_nxt_waypoints == False): # will not be connect at any of the next possible waypoints

                            ##if no route at next waypoint select waypoint towards current routes to bs nexthop node's next waypoint.
                            bs_idx, BS_connected_flag_now = self.connected_to_BS( G_now, uavID )
                            if BS_connected_flag_now:
                                nxt_hop=nx.shortest_path(G_now, source=uavID, target=self.bs_iDs[bs_idx])[1]
                                # print(nxt_hop, nx.shortest_path(G_aftertisec, source=uavID, target=self.bs_iD))
                                # print("nobreak---", np.linalg.norm(uav_pos_now[0:2,uavID]-self.bs_position))

                                if nxt_hop == self.bs_iDs[bs_idx]: #next hop is BS itself
                                    # Move next waypoint closest to BS
                                    bs_list = np.zeros(5)
                                    bs_list[np.argmin(dist_to_bs)] = 1
                                    Pi = bs_list
                                else:  #next hop is a  neighbor with connection to BS
                                    #move to  this neighbor next waypoint
                                    dist_to_nexthopnode = []
                                    for direction_R in waypoint_directions:
                                        nxt_waypoint = [None, None]
                                        nxt_waypoint[0], nxt_waypoint[1] = self.return_next_dst_point_based_on_direction( curr_cell, direction_R, self.hop_dist)

                                        # d_nxthop=np.sqrt( np.sum( np.square(arr(uav_pos_now[0:2,nxt_hop])-arr(nxt_waypoint[0:2])) )).astype(float)  # next hop node current position
                                        nxthopnode_nextwaypoint = msgs[8:10, nxt_hop]
                                        d_nxthop = np.sqrt( np.sum( np.square(arr(nxthopnode_nextwaypoint)-arr(nxt_waypoint[0:2])) )).astype(float)  # # next hop node next waypoint position
                                        dist_to_nexthopnode.append(d_nxthop)

                                    nxthop_list = np.zeros(5)
                                    nxthop_list[np.argmin(dist_to_nexthopnode)] = 1
                                    Pi = nxthop_list
                        else:
                            Pi = Pi * bs_connected_at_nxt_waypoints   # only consider BS connected waypoints
                    else:
                        raise NameError('Selected wrong SELECT MOBILILTY_MODEL_TYPE; choose CAP or BSCAP')


                    current_action = np.where(Pi == max(Pi))       # select waypoint/ heading direction with max Pi value
                    current_action = current_action[0]
                    current_action = np.random.choice(current_action)
                    direction_R = waypoint_directions[ current_action ]

    #                  ##########################################################################################################################
                    # get wayponit of selected next-wayoint cell center
                    controller[11], controller[12] = self.return_next_dst_point_based_on_direction( curr_cell, direction_R, self.hop_dist)

                    return current_state, current_action

                else: #% pheromone/rand model (NOT USED DURING TRAINING)
                    raise NameError('Not supposed to be here during training run')

        else: #% Something went wrong
            #otherwise
            raise NameError('[Agent %d] I am in an unknown state. Help!', controller[7])

        return None, None


    def current_shortest_hoplength_to_BS(self, G_now, uavID):  #BSedit
        hop_lengths = []
        for i in range(len(self.bs_iDs)):
            try:
                shortesthoplen_bs = nx.shortest_path_length(G_now, source=uavID, target=self.bs_iDs[i])
                hop_lengths.append((shortesthoplen_bs, i))
            except:
                continue
        if hop_lengths:
            idx = hop_lengths[0][1]
            min_hop = hop_lengths[0][0]
            for j in range(1, len(hop_lengths)):
                if hop_lengths[j][0] < min_hop:
                    min_hop = hop_lengths[j][0]
                    idx = hop_lengths[j][1]
            return idx, min_hop
        else:
            return -1, -1 # no path of BS, hop length is set to -1

    @profile
    def base_station_connected(self, uavID, msgs, nxt_waypoint ):  #BSedit
        # BS scheme FUNCTION

        uav_pos_next = np.copy(msgs[8:10,:])
        uav_pos_next[0:2, uavID] = nxt_waypoint
        # print("uav_pos_next", uav_pos_next[0:2, uavID])
        connMat_next = self.connectivity(uav_pos_next,self.transmission_range)
        G_next = nx.from_numpy_array(connMat_next)
        id, BS_connected_flag_next = self.connected_to_BS( G_next, uavID )

        if BS_connected_flag_next: # if connected to BS after ts sec
            return id, True
        else:
            return -1, False


    def connected_to_BS(self, G_now, uavID ):
        bs_id, shortesthoplen_bs = self.current_shortest_hoplength_to_BS(G_now, uavID)
        if shortesthoplen_bs==-1:
            return -1, False
        else:
            return bs_id, True


    def alpha_func_CAP(self, deg):  # distance-weighted-degree-of-connectivity
        if deg < self.alpha_type:
            alp = deg / self.alpha_type
        else:
            alp = 1.

        return alp

    def alpha_func_BSCAP(self, deg):  # distance-weighted-degree-of-connectivity
        if deg <= self.alpha_type:
            alp = deg / self.alpha_type

        elif self.alpha_type < deg <=3:
            alp = 1.

        elif 3 < deg <=5:
            alp = 1. - (1./3)*(deg -3)

        else:
            alp = 1./3

        return alp



    #% Are we too close to another agent?
    @profile
    def airspace(self, controller, nextpos, msgs, dt, uav_airspace):
        stop = False;

        them = 0;
        index = 0;
        closest = np.inf;

        # ATI: do a simple thresholding before running through the messages, based on current position etc.
        # Only check messages (& estimate projected position) from UAVs that are already "close enough"
        init_dist2 = ((nextpos[0:2,np.newaxis]-msgs[0:2,:])**2).sum(0)
        check = np.where( init_dist2 < 16*(self.uav_airspace+self.__boundv(dt)[1])**2 )[0]

        for jj in check: #range(len(msgs)):
            #% Get the other agent, skipping ourself
            if jj == controller[7]:
                continue
            other = msgs[:,jj]; #%msgs{jj};

            #% If it continued, where would it be?
            #% SHREY: estimate the position of the neighbor agents from their previous 5 positions
            x, y, theta = rk4(other[0], other[1], other[3], other[5], other[6], dt);

            #% Too close to our projected location?
            dist = ((nextpos[0:2]-[x,y])**2).sum().round();    #% SHREY: nextpos(x,y) is my future position estimate
            # if dist < closest and dist < 2*self.uav_airspace:  # ATI: added 2nd condition vvv
            if dist < closest and dist < 2*(self.uav_airspace**2):
                them = arr([x,y,theta]);
                dx = nextpos[0]-them[0];
                dy = nextpos[1]-them[1];
                next_v = arr([dx,dy]);
    #          next_v = (next_v/np.linalg.norm(next_v) )* 60;
                next_v = ( next_v / np.sqrt(np.sum(next_v**2)) )* 60; # Note next_v vector should not be zero;
                xa = nextpos[0]-controller[0];
                ya = nextpos[1]-controller[1];

                push_v = arr([xa,ya]);
    #          push_v = (push_v/np.linalg.norm(push_v) )* 60;
                push_v = (push_v / np.sqrt(np.sum(push_v**2)) )* 60;  # Note push_v vector should not be zero;

                combined_v = np.add(next_v,push_v); #combined_v = next_v+push_v;
                controller[11] = controller[0]+combined_v[0];
                controller[12] = controller[1]+combined_v[1];
        #          plot(controller[11],controller[12],'bx');

                controller[13] = 0;
                stop = True;

        return controller,stop



    #% Too close?
    @profile
    def proximity(self, controller, nextpos, them, index, closest, radius): #%,   nx2 ,ot2):
        stop = False;

        #% Is there a closest neighbour?
        if closest < np.inf:
            #% If same neighbour as before & further away, leave as-is
            if index == controller[9]:
                if closest > controller[10]:
                    return controller, stop
            dx = nextpos[0]-them[0];
            dy = nextpos[1]-them[1];
            next_v = arr([dx,dy]);
    #          next_v = (next_v/np.linalg.norm(next_v) )* 60;
            next_v = ( next_v / np.sqrt(np.sum(next_v**2)) )* 60; # Note next_v vector should not be zero;

            xa = nextpos[0]-controller[0];
            ya = nextpos[1]-controller[1];

            push_v = arr([xa,ya]);
    #          push_v = (push_v/np.linalg.norm(push_v) )* 60;
            push_v = (push_v / np.sqrt(np.sum(push_v**2)) )* 60;  # Note push_v vector should not be zero;

            combined_v = np.add(next_v,push_v); #combined_v = next_v+push_v;
            controller[11] = controller[0]+combined_v[0];
            controller[12] = controller[1]+combined_v[1];
    #          plot(controller[11],controller[12],'bx');

            controller[13] = 0;
            stop = True;

        return controller,stop

    #% Turn towards coordinates
    def face(self, controller, x, y, dt):
        delta = (180/np.pi)*np.arctan2(x-controller[0],y-controller[1]) - controller[3];  #SHREY: atan2d returns angle between [-180, 180] degerees
        controller, steps = self.turn(controller, delta, dt);
        return controller,steps

    #% Picks a random target
    def randomtarg(self, controller, lowerX, upperX, lowerY, upperY):
        #% Pick a random location, between bounds
        a = max(lowerX,0);
        b = min(upperX,self.map_size);
        r = (b-a)*np.random.rand() + a;
        controller[11] = round(r);
        a = max(lowerY,0);
        b = min(upperY,self.map_size);
        r = (b-a)*np.random.rand() + a;
        controller[12] = round(r);

        #%     angle=controller(4); %rand*360;
        #%     xdot = 40*sind(angle)
        #%     ydot = 40*cosd(angle)
        #%     controller(12)=(controller(1) + xdot)
        #%     controller(13)=(controller(2) + ydot)
        return controller

    #% Turn by delta degrees
    def turn(self, controller, delta, dt):
        b = self.__boundv(dt);                  # Make bounds object
        delta = (delta+180)%360 - 180;          # Normalise the angle

        if delta == 0:   # Nothing to do?
            steps = 1;
            return controller,steps

        #% Delta is now in deg/s
        delta = delta / (1*dt);  #% SHREY : multiplying by 2, 3.. for mor steps and reduce max turn in bounds.m

        #% Something to do!
        steps = np.ceil( abs(delta) / b[3] / b[1] );     # ATI closed form calculation
        v = max(np.ceil( abs(delta) / steps / b[3]), b[0]);
        mu = delta / steps / v;            # find integer s,v : b(1) < v < b(2), b(3) < mu < b(4) & delta = mu*v*s

        controller[5] = v;
        controller[6] = mu;
        return controller,steps


    def move(self, robot, v, mu, dt): # velcoity and and angle obtained using Runge-Kutta (rk4)
        # Make bounds object
        b = self.__boundv(dt);

        v = min(max(v,b[0]),b[1]);    # Physically cap speed
        mu = min(max(mu,b[2]),b[3]);  # Physically cap turn

        # Runge-Kutta (rk4)
        robot[0], robot[1], robot[3] = rk4(robot[0], robot[1], robot[3], v, mu, dt);
        return robot


    def mod_circcirc_octave(self, x1,y1,r1,x2,y2,r2):
        P1 = arr([x1,y1]);
        P2 = arr([x2,y2]);
        d2 = sum((P2-P1)**2);

        P0 = (P1+P2)/2+(r1**2-r2**2)/d2/2*(P2-P1);
        t = ((r1+r2)**2-d2)*(d2-(r2-r1)**2);
        if t <= 0:
            #print("The circles don't intersect.\n")
            xout=arr([np.nan, np.nan]);
            yout=arr([np.nan, np.nan]);
        else:
            T = np.sqrt(t)/d2/2*arr([ [0,-1],[1,0]]).dot(P2-P1);
            Pa = P0 + T; #% Pa and Pb are circles' intersection points
            Pb = P0 - T;

            xout=arr([Pa[0], Pb[0]]);
            yout=arr([Pa[1], Pb[1]]);
        return (xout,yout)


    """
    function [xout,yout]=circcirc_FAST(x1,y1,r1,x2,y2,r2)
        %CIRCCIRC  Intersections of circles in Cartesian plane
        %
        %  [xout,yout] = CIRCCIRC(x1,y1,r1,x2,y2,r2) finds the points
        %  of intersection (if any), given two circles, each defined by center
        %  and radius in x-y coordinates.  In general, two points are
        %  returned.  When the circles do not intersect or are identical,
        %  NaNs are returned.  When the two circles are tangent, two identical
        %  points are returned.  All inputs must be scalars.
        %
        %  See also LINECIRC.

        % Copyright 1996-2007 The MathWorks, Inc.
        % $Revision: 1.10.4.4 $    $Date: 2007/11/26 20:35:08 $
        % Written by:  E. Brown, E. Byrns

        r3=sqrt((x2-x1).^2+(y2-y1).^2);

        indx1=find(r3>r1+r2);  % too far apart to intersect
        indx2=find(r2>r3+r1);  % circle one completely inside circle two
        indx3=find(r1>r3+r2);  % circle two completely inside circle one
        indx4=find((r3<10*eps)&(abs(r1-r2)<10*eps)); % circles identical
        indx=[indx1(:);indx2(:);indx3(:);indx4(:)];

        anought=atan2((y2-y1),(x2-x1));

        %Law of cosines

        aone=acos(-((r2.^2-r1.^2-r3.^2)./(2*r1.*r3)));

        alpha1=anought+aone;
        alpha2=anought-aone;

        xout=[x1 x1]+[r1 r1].*cos([alpha1 alpha2]);
        yout=[y1 y1]+[r1 r1].*sin([alpha1 alpha2]);

        % Replace complex results (no intersection or identical)
        % with NaNs.

        if ~isempty(indx)
            xout(indx,:) = NaN;
            yout(indx,:) = NaN;
        end
    end
    """


    #% gets the pheromone value of neighbouring cells
    def get_neighbor_pheromone_weight(self, node_id,curr_cell):
        """Extract neighboring cell pheromone information"""

        if (self.hop_dist==1): H = arr([[1]]);            # define filter for pheromones
        else: H = (1./12)*arr([[1.,1.,1.],[1.,4.,1.],[1.,1.,1.]]);   # sIMPLIED FOR TRAINING A SIMPLE Q-MODEL    LATER:??? CHANGE
        #   H = arr([[1]]) ??
        W = int(self.hop_dist + (max(H.shape)+1)/2) #W = int(self.hop_dist * (max(H.shape)+1)/2);        # get filter maximum width
        buffer_value = 4                              # "beyond edge" values

        extracted = np.zeros((2*W+1,2*W+1)) + buffer_value;  # store extracted sub-map
        N0,N1,_ = self.node_pheromone_map.shape
        i,j = int(curr_cell[0]), int(curr_cell[1])
        i0,i1 = max(i-W,0), min(i+W+1,N0); ii0 = i0-i+W; ii1 = 2*W+1 +i1-i-W-1
        j0,j1 = max(j-W,0), min(j+W+1,N1); jj0 = j0-j+W; jj1 = 2*W+1 +j1-j-W-1
        try: extracted[ii0:ii1,jj0:jj1] = self.node_pheromone_map[i0:i1,j0:j1,int(node_id)]
        except ValueError: print(curr_cell,"\n"); print(ii0,ii1,jj0,jj1," <- ",i0,i1,j0,j1)

        #   temp = convolve2d( extracted, H, boundary='symm', mode='same') # temp = convolve2d(extracted,H,'same');
        temp = convolveim( extracted, H, mode='constant') # temp = convolve2d(extracted,H,'same');

        weight = temp[ W+self.hop_dist*arr([0,1,1,1,0,-1,-1,-1]), W+self.hop_dist*arr([1,1,0,-1,-1,-1,0,1]) ]

        #if np.any(weight>1):
        # raise ValueError("get_neighbor_pheromone_weight return value >1")
        return weight


    def return_next_dst_point_based_on_direction(self, curr_cell, R, hop_dist):
        # controller_target_x, controller_target_y = None,None
        # ATI replacement code
        #      sub = 1 if hop_dist==5 else 0;           # what is this?  % SHREY: BUG_FIX commented uwanted lines
        #      if (hop_dist % 2 == 1): hop_dist -= sub; # ^^ ???
        xoff = [0,1,1,1,0,-1,-1,-1];
        yoff = [1,1,0,-1,-1,-1,0,1];
        controller_target = curr_cell + arr([xoff[R],yoff[R]])*hop_dist
        #####controller_target = (controller_target + 1 - .5) * self.map_resolution # SHREY: BUG_FIX replaced controller_target = (controller_target - .5) * self.map_resolution
        controller_target = (controller_target - .5) * self.map_resolution
        #print(controller_target)
        return controller_target

    # function merge pheromone map of connected UAV neighbors
    def merge_pheromone_map(self, uavID, nbrs):
        self.node_pheromone_Repel[:,:,nbrs] = np.maximum(self.node_pheromone_Repel[:,:,nbrs],self.node_pheromone_Repel[:,:,uavID:uavID+1])

        # removed additional code (commented)


 
    def plot_UAV_figures(self, t, tidx, ax, axx, drawplots, plot_interval, drawPheromonemap, drawUAVflight, drawAirspace, drawUAVconnectivity, connMat):  # BSedit
        if (drawplots and ((t % plot_interval) == 0)):
            # plt.cla()
            plt.clf()

            # Make colour
            color = self.colors[np.arange(self.nAgents).astype(int) % self.colors.shape[0], :]

            if drawUAVflight:
                # plot robot locations
                plt.scatter(self.Arobot[0, :]/self.map_resolution, self.Arobot[1, :]/self.map_resolution, c=color)
                plt.axis([0, self.map_size/self.map_resolution, 0, self.map_size/self.map_resolution])
                for uavID in range(self.nAgents):
                    plt.text(self.Arobot[0, uavID]/self.map_resolution, self.Arobot[1,uavID]/self.map_resolution, str(uavID))

                plt.grid(True)
                # for uavID in range(self.nAgents):
                #     xyz = np.vstack((self.Arobot_history[tidx % self.nHistory+1:, :, uavID],
                #                     self.Arobot_history[:tidx % self.nHistory, :, uavID], self.Arobot[np.newaxis, 0:3, uavID]))
                #     plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], ':', c=color[uavID, :])
                plt.plot(self.Acontroller[11, :]/self.map_resolution, self.Acontroller[12, :]/self.map_resolution, 'cx')

            rr, cc = np.nonzero(connMat)
            for i, j in zip(rr, cc):
                plt.plot([self.Arobot[0, i]/self.map_resolution, self.Arobot[0, j]/self.map_resolution], [self.Arobot[1, i]/self.map_resolution, self.Arobot[1, j]/self.map_resolution],
                          'b-', lw=.5)  # BSedit
            if drawPheromonemap:
                plt.imshow(self.node_pheromone_map[:, :, 0].T, vmin=0, vmax=1)
                plt.colorbar()
            plt.title('Simulation state at t={} secs'.format(t))
            plt.pause(.01)

            
            
@jit(nopython=True)
def warshall(adj_matrix):
    n = adj_matrix.shape[0]
    closure = adj_matrix.astype(np.bool_)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                closure[i, j] = closure[i, j] or (closure[i, k] and closure[k, j])
    
    return closure
          

rk4 = None

def rk4_py(x, y, theta, v, mu, dt):
    """ Runge-Kutta (rk4) """

    def f_continuous(theta, v, mu):
        xdot = v*np.sin(theta);
        ydot = v*np.cos(theta);
        thetadot = v*mu;
        return xdot,ydot,thetadot

    thetar = theta/180*np.pi;
    mur = mu/180*np.pi;
    k1_x, k1_y, k1_theta = f_continuous(thetar, v, mur);
    k2_x, k2_y, k2_theta = f_continuous(thetar + k1_theta*dt/2, v, mur);
    k3_x, k3_y, k3_theta = f_continuous(thetar + k2_theta*dt/2, v, mur);
    k4_x, k4_y, k4_theta = f_continuous(thetar + k3_theta*dt, v, mur);

    x = x + (k1_x+2*k2_x+2*k3_x+k4_x)*dt/6;
    y = y + (k1_y+2*k2_y+2*k3_y+k4_y)*dt/6;
    theta = theta + 180/np.pi*(k1_theta+2*k2_theta+2*k3_theta+k4_theta)*dt/6;

    theta = theta % 360;
    return x,y,theta

try:
    from phero_c import rk4 as rk4_cy
    rk4 = rk4_cy
except:
    print("pheromone.py: error loading cython file; defining in pure python (slower)")
    print("  to build, run:  python3 setup.py build_ext --inplace ");
    rk4 = rk4_py
