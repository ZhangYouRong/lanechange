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
import agent
import numpy as np


class Env:
    TOWN = 'Town04'
    VEHICLE_TYPE = 'vehicle.tesla.model3'
    VEHICLE_COLOR_RGB = '255,255,255'
    VEHICLE_START_LOCATION = {'x': -9.8, 'y': -186.6, 'z': 0}  # 我是通过manual_control.py手动测的坐标
    PIDCONTROLLER_TIME_PERIOD = 0.05  # 0.05s

    observation_space_dim = 2  # 横向误差和航向误差
    action_dim = 1  # steer
    max_action = 1  # max steering angle
    action_space_low = -1
    action_space_high = 1

    def __init__(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(10)
        self.world = client.load_world(self.TOWN)

        spectator = self.world.get_spectator()
        spectator_location = carla.Location(x=self.VEHICLE_START_LOCATION['x'],
                                            y=self.VEHICLE_START_LOCATION['y'],
                                            z=self.VEHICLE_START_LOCATION['z']+7)
        spectator_rotation = carla.Rotation(yaw=90)
        spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
        # disable all graphic rendering
        print('\nenabling synchronous & no rendering mode.')
        settings = self.world.get_settings()
        settings.no_rendering_mode = True
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        blueprint_library = self.world.get_blueprint_library()
        self.bp = blueprint_library.find(self.VEHICLE_TYPE)
        self.bp.set_attribute('color', self.VEHICLE_COLOR_RGB)

        self.map = self.world.get_map()
        start_waypoint = self.map.get_waypoint(carla.Location(x=self.VEHICLE_START_LOCATION['x'],
                                                              y=self.VEHICLE_START_LOCATION['y'],
                                                              z=self.VEHICLE_START_LOCATION['z']))
        self.spawn_point = carla.Transform(start_waypoint.transform.location,
                                           start_waypoint.transform.rotation)
        # self.spawn_point.location.z += 0.015  # without this may lead to collision error for spawn operation
        self.actor_list = []
        self.lane_change_agent = None
        self.reset()

    def reset(self):
        if self.actor_list is not None:
            for actor in self.actor_list:
                actor.destroy()
        if self.lane_change_agent is not None:
            del self.lane_change_agent

        self.actor_list = []
        self.vehicle = self.world.spawn_actor(self.bp, self.spawn_point)
        self.actor_list.append(self.vehicle)

        self.world.tick()
        tick = self.world.wait_for_tick()
        self.episode_start_time = tick.elapsed_seconds
        self.simulation_time = 0
        self.lane_change_agent = agent.Agent(self.vehicle, self.PIDCONTROLLER_TIME_PERIOD)
        self.lane_change_agent.get_current_data(self.simulation_time)
        while self.simulation_time < 2.5:
            next_state, _, _, _ = self.step([0])  # 去除一开始没加速的S,A,R,S'数据
        # next_state, _, _, _ = self.step([0])  # 去除一开始没加速的S,A,R,S'数据
        return next_state

    def step(self, action):
        control, finish = self.lane_change_agent.run_step()
        control.steer = action[0]
        self.vehicle.apply_control(control)

        self.world.tick()
        tick = self.world.wait_for_tick()
        self.simulation_time = tick.elapsed_seconds-self.episode_start_time

        next_state = self.lane_change_agent.get_current_data(self.simulation_time)

        # if self.lane_change_agent.change_times == 0 and self.simulation_time > 7:
        #     self.lane_change_agent.lane_change_flag = agent.RoadOption.CHANGELANELEFT

        self.data = np.array(self.lane_change_agent.data).transpose()
        reward = -(0.2*(self.data[3][-1]/(0.3*9.8))**2*self.PIDCONTROLLER_TIME_PERIOD \
                   +0.8*(self.data[5][-1]/3.5)**2*self.PIDCONTROLLER_TIME_PERIOD)

        # print('K:%f'%K, 'L:%f'%L,
        #       'Time spend:%f'%lane_change_agent.lane_change_duration, 'J:%f'%J)
        fail = 0

        if abs(next_state[-1])*agent.DELTA_FI_RANGE > 60:
            fail = 1  # 撞击使速度<5,或直接掉头,或越出道路
            reward -= 0.5
        if abs(next_state[0]*agent.D_LATERAL_RANGE) > 4.5:
            fail = 1  # 撞击使速度<5,或直接掉头,或越出道路
            reward -= 0.1
        # info = 0
        # if self.lane_change_agent.lane_change_duration is not None:  # 换道结束
        #     info = 1
        return next_state, reward, fail, finish

    def render_mode(self):
        settings = self.world.get_settings()
        settings.no_rendering_mode = False
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def __del__(self):
        print('\ndisabling synchronous mode.')
        settings = self.world.get_settings()
        settings.no_rendering_mode = True
        settings.synchronous_mode = False
        self.world.apply_settings(settings)