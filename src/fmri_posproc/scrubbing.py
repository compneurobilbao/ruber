#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:23:57 2017

@author: asier
"""


def scrubbing(time_series, FD, thres=0.2):
    """
    simple scrubbing strategy based on timepoint removal
    """
    
    scrubbed_time_series = time_series.T[:,(FD<thres)]
    
    return scrubbed_time_series.T
