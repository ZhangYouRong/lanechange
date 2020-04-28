# must run the server in stable FPS=20!
# for windows, enter this in the command line (in the root directory of carla)
# D:\software\CARLA_0.9.5\CarlaUE4.exe -benchmark -FPS=20

import glob
import os
import sys

try:
    sys.path.append(glob.glob('D:\software\CARLA_0.9.5\PythonAPI\carla\dist\carla-*%d.%d-%s.egg'%(
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import math
from agent import Agent, RoadOption
import agent
import misc
import controller

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def plot_data(data,lane_change_agent):
    fontdict = {'weight': 'normal', 'size': 20}
    plt.figure(figsize=(16, 9))
    ax1 = plt.subplot(3, 2, 1)  # （行，列，活跃区）
    plt.plot(data[0], data[1], color='b')
    plt.xlabel('t (s)', fontdict)
    plt.ylabel('Volocity (km/h)', fontdict)

    ax2 = plt.subplot(3, 2, 2)
    plt.plot(data[0], data[2], color='g')
    plt.xlabel('t (s)', fontdict)
    plt.ylabel('steer (-1,1)', fontdict)

    ax3 = plt.subplot(3, 2, 3)
    plt.plot(data[0], data[3], color='y')
    plt.xlabel('t (s)', fontdict)
    plt.ylabel('a$_{lateral}$ (m/s$^2$)', fontdict)

    ax4 = plt.subplot(3, 2, 4)
    plt.plot(data[0], data[4], color='c')
    plt.xlabel('t (s)', fontdict)
    plt.ylabel('jerk (m/s$^3$)', fontdict)

    ax5 = plt.subplot(3, 2, 5)
    plt.plot(data[0], data[5], color='r')
    plt.xlabel('t (s)', fontdict)
    plt.ylabel('d$_{lateral}$ (m)', fontdict)

    angle = lane_change_agent.trajectory_rotation_angle
    x = data[6]*math.cos(angle)+data[7]*math.sin(angle)
    y = data[6]*-math.sin(angle)+data[7]*math.cos(angle)
    delta_x = x-x[0]
    delta_y = y-y[0]

    ax5 = plt.subplot(3, 2, 6)
    plt.plot(delta_x, delta_y, color='m')
    plt.xlabel('x (m)', fontdict)
    plt.ylabel('y (m)', fontdict)
    plt.show()

