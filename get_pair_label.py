#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 21:54:15 2017

@author: xfz
"""
import numpy as np

def getPairLabel(labels):
    
    shape = len(labels)
    pairL = np.zeros((shape,shape))
    for i,l in enumerate( labels):
        pairL[i]=np.equal(labels,l)
    return pairL
    
    
    