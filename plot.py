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

import csv
import codecs


def plot_heatmap(data, xtick, ytick):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks(np.linspace(0, 120, 7), np.linspace(-60, 60, 7))
    plt.yticks(np.linspace(0, 90, 5), np.linspace(-4.5, 4.5, 5))
    # ax.set_xticks(range(len(tick1)))
    # ax.set_yticks(range(len(tick2)))

    fontdict = {'weight': 'normal', 'size': 16}
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    im = ax.imshow(data)
    plt.xlabel("航向误差$\phi$ (°)", fontdict)
    plt.ylabel("横向误差e (m)", fontdict)
    plt.colorbar(im)
    plt.show()


def plot_data(data, lane_change_agent):
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


def plot_data_essay(data, lane_change_agent):
    fontdict = {'weight': 'normal', 'size': 20}
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.plot(data[0], data[1], color='b')
    plt.xlabel('t (s)', fontdict)
    plt.ylabel('Volocity (km/h)', fontdict)
    plt.title('车速V$_{0}$随时间t变化图')
    plt.show()

    # plt.figure(figsize=(12, 9))
    # ax2 = plt.subplot(2, 2, 1)
    # ax2.set_title('方向盘转角s随时间t变化图')
    plt.plot(data[0], data[2], color='g')
    plt.xlabel('t (s)', fontdict)
    plt.ylabel('steer (-1,1)', fontdict)
    plt.show()

    # ax3 = plt.subplot(2, 2, 2)
    # ax3.set_title('侧向加速度a$_{y}$随时间t变化图')
    plt.plot(data[0], data[3], color='y')
    plt.xlabel('t (s)', fontdict)
    plt.ylabel('a$_{y}$ (m/s$^2$)', fontdict)
    plt.show()

    # ax4 = plt.subplot(3, 2, 4)
    # plt.plot(data[0], data[4], color='c')
    # plt.xlabel('t (s)', fontdict)
    # plt.ylabel('jerk (m/s$^3$)', fontdict)

    # ax5 = plt.subplot(2, 2, 3)
    # ax5.set_title('侧向误差e随时间t变化图')
    plt.plot(data[0], data[5], color='r')
    plt.xlabel('t (s)', fontdict)
    plt.ylabel('e (m)', fontdict)
    plt.show()

    angle = lane_change_agent.trajectory_rotation_angle
    x = data[6]*math.cos(angle)+data[7]*math.sin(angle)
    y = data[6]*-math.sin(angle)+data[7]*math.cos(angle)
    delta_x = x-x[0]
    delta_y = y-y[0]

    # ax6 = plt.subplot(2, 2, 4)
    # ax6.set_title('车辆实际运动路径图')
    plt.plot(delta_x, delta_y, color='m')
    plt.xlabel('x (m)', fontdict)
    plt.ylabel('y (m)', fontdict)
    plt.show()

    save_data = [delta_x, delta_y]
    data_write_csv("D:\software\CARLA_0.9.5\workspace\lanechange\exp2020-04-22-10-33-48\ddpg.csv", save_data)


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")
