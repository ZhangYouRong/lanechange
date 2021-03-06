# cd C:\Program Files\Carla
# CarlaUE4.exe -benchmark -FPS=20
# tensorboard --logdir=D:\software\CARLA_0.9.5\workspace\lanechange\
# http://localhost:6006

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

TOWN = 'Town04'
VEHICLE_TYPE = 'vehicle.tesla.model3'
VEHICLE_COLOR_RGB = '255,255,255'
VEHICLE_START_LOCATION = {'x': -9.8, 'y': -186.6, 'z': 0}  # 我是通过manual_control.py手动测的坐标
PIDCONTROLLER_TIME_PERIOD = 0.05  # 0.05s


def plot_data(data):
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
    plt.plot(delta_x, delta_y, color='b')
    data = np.load('D:\software\CARLA_0.9.5\workspace\lanechange\exp2020-04-22-10-33-48\ddpg.npy')
    plt.plot(data[1],data[0],color='r')
    label = ["传统算法路径", "强化学习优化路径"]
    plt.legend(label,fontsize=16)
    plt.xlabel('x (m)', fontdict)
    plt.ylabel('y (m)', fontdict)
    plt.show()


if __name__ == '__main__':

    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    world = client.load_world(TOWN)
    spectator = world.get_spectator()
    spectator_location = carla.Location(x=VEHICLE_START_LOCATION['x'],
                                        y=VEHICLE_START_LOCATION['y'],
                                        z=VEHICLE_START_LOCATION['z']+7)
    spectator_rotation = carla.Rotation(yaw=90)
    spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

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
    actor_list = []
    map = world.get_map()
    start_waypoint = map.get_waypoint(carla.Location(x=VEHICLE_START_LOCATION['x'],
                                                     y=VEHICLE_START_LOCATION['y'],
                                                     z=VEHICLE_START_LOCATION['z']))
    spawn_point = carla.Transform(start_waypoint.transform.location, start_waypoint.transform.rotation)

    # episode circulation
    vehicle = world.spawn_actor(bp, spawn_point)
    actor_list.append(vehicle)
    print('created %s'%vehicle.type_id)

    world.tick()
    world_snapshot = world.wait_for_tick()
    start_world_time = world_snapshot.elapsed_seconds
    simulation_time = 0
    lane_change_agent = Agent(vehicle, PIDCONTROLLER_TIME_PERIOD)
    lane_change_agent.run_step(simulation_time)
    finish = False

    while not finish:  # 完成后跳出循环
        world.tick()  # Initialize a new "tick" in the simulator.
        world_snapshot = world.wait_for_tick()  # Wait until we listen to the new tick.
        simulation_time = world_snapshot.elapsed_seconds-start_world_time
        finish = lane_change_agent.run_step(simulation_time)
        # apply control and get data

    data = np.array(lane_change_agent.data).transpose()
    J = 0.2*sum((data[3]/(0.3*9.8))**2*PIDCONTROLLER_TIME_PERIOD) \
        +0.8*sum((data[5]/(3.5))**2*PIDCONTROLLER_TIME_PERIOD)
    print('Time spend on lane change:%f'%lane_change_agent.lane_change_duration)
    print('J:%f'%J)
    plot_data(data)

    # destroy
    print('\ndisabling synchronous mode.')
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    for actor in actor_list:
        actor.destroy()
        print("All cleaned up!")
