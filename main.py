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
import time
import random
import math
from agent import Agent, RoadOption
import agent
import misc
import controller

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

TOWN = 'Town03'
VEHICLE_TYPE = 'vehicle.tesla.model3'
VEHICLE_COLOR_RGB = '255,255,255'
VEHICLE_START_LOCATION = {'x': 46, 'y': 7.2, 'z': 0}  # 我是通过manual_control.py手动测的坐标
PIDCONTROLLER_TIME_PERIOD = 0.05  # 0.05s
KLIST = np.linspace(43/540, 53/540, 11)
LLIST = np.linspace(0.10, 0.14, 9)


# 49/540 0.0115

def plot_data(data):
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


if __name__ == '__main__':

    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    world = client.load_world(TOWN)
    # disable all graphic rendering
    print('\nenabling synchronous & no rendering mode.')
    settings = world.get_settings()
    settings.no_rendering_mode = True
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.find(VEHICLE_TYPE)
    bp.set_attribute('color', VEHICLE_COLOR_RGB)

    map = world.get_map()
    start_waypoint = map.get_waypoint(carla.Location(x=VEHICLE_START_LOCATION['x'],
                                                     y=VEHICLE_START_LOCATION['y'],
                                                     z=VEHICLE_START_LOCATION['z']))
    spawn_point = carla.Transform(start_waypoint.transform.location, start_waypoint.transform.rotation)

    a = np.zeros((len(KLIST), len(LLIST)))

    for i, K in enumerate(KLIST):
        for j, L in enumerate(LLIST):
            agent.args_lateral_dict['K'] = K
            agent.args_lateral_dict['Lookahead_Distance'] = L
            # episode circulation
            vehicle = world.spawn_actor(bp, spawn_point)
            actor_list = []
            actor_list.append(vehicle)
            # print('created %s'%vehicle.type_id)

            world.tick()
            tick = world.wait_for_tick()
            start_world_time = tick.elapsed_seconds
            simulation_time = 0
            lane_change_agent = Agent(vehicle, PIDCONTROLLER_TIME_PERIOD)
            lane_change_agent.run_step(simulation_time)

            while simulation_time < 25:  # 25s后跳出循环
                world.tick()  # Initialize a new "tick" in the simulator.
                tick = world.wait_for_tick()  # Wait until we listen to the new tick.
                simulation_time = tick.elapsed_seconds-start_world_time
                lane_change_agent.run_step(simulation_time)
                # apply control and get data

                if lane_change_agent.change_times == 0 and simulation_time > 7:
                    lane_change_agent.lane_change_flag = RoadOption.CHANGELANELEFT

            data = np.array(lane_change_agent.data).transpose()
            J = 0.2*sum((data[3]/(0.3*9.8))**2*PIDCONTROLLER_TIME_PERIOD) \
                +0.8*sum((data[5]/(3.5))**2*PIDCONTROLLER_TIME_PERIOD)
            print('K:%f'%K, 'L:%f'%L,
                  'Time spend:%f'%lane_change_agent.lane_change_duration, 'J:%f'%J)

            a[i][j] = J
            # plot_data(data)

            # destroy
            del lane_change_agent
            for actor in actor_list:
                actor.destroy()

    print('\ndisabling synchronous mode.')
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x, y = np.meshgrid(LLIST, KLIST)

    ax.plot_surface(x, y, a, cmap='rainbow')
    ax.set_xlabel('K')
    ax.set_ylabel('L',)
    ax.set_zlabel('J')
    plt.show()
    print(a.min())
