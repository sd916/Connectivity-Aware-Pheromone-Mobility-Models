#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:12:06 2020

@author: shreyasdevaraju
"""
# globals.py caontains all the global vaiables such as the Q-table and episode counter



import numpy as np

def init():
    global connectivity_histogram

    connectivity_histogram = np.zeros(11)
