import carla
import misc
import controller
import schedule
import time
import math
import random
import numpy as np
from enum import Enum
from collections import deque

args_lateral_dict = {
    'K_P': 1,
    'K_D': 0.02,
    'K_I': 0,
    'dt': 1.0/20.0,
    # distance method
    'K': 20/540,
    'Lookahead_Distance': 0.3,
}
args_longitudinal_dict = {
    'K_P': 0.1,
    'K_D': 0.01,
    'K_I': 0.05,
    'dt': 1.0/20.0}
TARGET_SPEED = 20
BIAS = 3.15
MIN_DISTANCE_PERCENTAGE = 0.9


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class Agent(object):
    def __init__(self, vehicle, pid_time_period=0.05):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()
        init_location=self._vehicle.get_location()
        self._current_waypoint = self._map.get_waypoint(init_location)
        self._sampling_radius = TARGET_SPEED/3.6
        self._min_distance = self._sampling_radius*MIN_DISTANCE_PERCENTAGE
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=4)
        self._buffer_size = 1
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))
        self._target_road_option = None
        self.lane_change_flag = RoadOption.LANEFOLLOW
        self.lane_change_start = None
        self.lane_change_duration = 0

        self.change_times = 0
        self.target_waypoint = self._current_waypoint

        self._pid_controller = controller.VehiclePIDController(vehicle,
                                                               args_lateral=args_lateral_dict,
                                                               args_longitudinal=args_longitudinal_dict)
        self._pid_timeperiod = pid_time_period

        self.x = None
        self.y = None
        self.yaw = None
        self.a_lateral_past = 0
        self.a_lateral = 0
        self.jerk = None
        self.d_lateral = None
        self.volocity = None

        self._control = carla.VehicleControl()
        self._control.steer = 0.0
        self._control.throttle = 0.0
        self._control.brake = 1.0
        self._control.hand_brake = False
        self._control.manual_gear_shift = False

        self.data = None

        self.trajectory_rotation_angle = math.radians(self._current_waypoint.transform.rotation.yaw)
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=1)

    def run_step(self, simulation_time):
        # not enough waypoints in the horizon? => add more!
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen*0.5):
            self._compute_next_waypoints(k=1, lane_change_flag=self.lane_change_flag)

        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        #   Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # target waypoint
        self.target_waypoint, self._target_road_option = self._waypoint_buffer[0]
        # move using PID controllers
        self._control = self._pid_controller.run_step(TARGET_SPEED+BIAS, self.target_waypoint)

        self._get_current_data(simulation_time)
        # print(len(self._waypoints_queue))

        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if misc.distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index+1):
                self._waypoint_buffer.popleft()

        self._vehicle.apply_control(self._control)

    def _get_current_data(self, simulation_time):
        transform = self._vehicle.get_transform()
        acceleration = self._vehicle.get_acceleration()

        location = transform.location

        self.x = location.x
        self.y = location.y
        self.yaw = transform.rotation.yaw
        self.a_lateral_past = self.a_lateral
        self.a_lateral = -acceleration.x*math.sin(math.radians(self.yaw))+acceleration.y*math.cos(
            math.radians(self.yaw))
        self.jerk = (self.a_lateral-self.a_lateral_past)*self._pid_timeperiod
        self.d_lateral = misc.distance_point_to_line(self.target_waypoint, transform)
        self.volocity = misc.get_speed(self._vehicle)

        print('x= %f'%self.x, 'y= %f'%self.y,
              'yaw= %f'%self.yaw, 'a_lateral= %f'%self.a_lateral,
              'jerk=%f'%self.jerk, 'simulation_time=%f'%simulation_time,
              'd_lateral= %f'%self.d_lateral, 'v= %f'%self.volocity)

        if self.lane_change_start is None and abs(self.d_lateral) > 2.5:
            self.lane_change_start = simulation_time
        if self.lane_change_start is not None and abs(self.d_lateral) < 0.25:
            self.lane_change_duration = simulation_time-self.lane_change_start

        if self.data is not None:
            self.data = self.data+[[simulation_time, self.volocity,
                                    self._control.steer, self.a_lateral,
                                    self.jerk, self.d_lateral,
                                    self.x, self.y]]
        else:
            self.data = [[simulation_time, self.volocity,
                          self._control.steer, self.a_lateral,
                          self.jerk, self.d_lateral,
                          self.x, self.y]]

    def _compute_next_waypoints(self, k=1, lane_change_flag=RoadOption.LANEFOLLOW):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        if lane_change_flag == RoadOption.LANEFOLLOW:
            # check we do not overflow the queue
            available_entries = self._waypoints_queue.maxlen-len(self._waypoints_queue)
            k = min(available_entries, k)

            for _ in range(k):
                last_waypoint = self._waypoints_queue[-1][0]
                next_waypoints = list(last_waypoint.next(self._sampling_radius))

                # 在路口，两条路中间就会出现next_waypoints>1的情况
                if len(next_waypoints) == 1:
                    # only one option available ==> lanefollowing
                    next_waypoint = next_waypoints[0]
                    road_option = RoadOption.LANEFOLLOW
                else:
                    # random choice between the possible options
                    road_options_list = _retrieve_options(
                        next_waypoints, last_waypoint)
                    road_option = random.choice(road_options_list)
                    next_waypoint = next_waypoints[road_options_list.index(
                        road_option)]

                self._waypoints_queue.append((next_waypoint, road_option))

        elif lane_change_flag == RoadOption.CHANGELANELEFT:
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoint = last_waypoint.get_left_lane().next(self._sampling_radius)[0]
            road_option = RoadOption.CHANGELANELEFT
            self._waypoints_queue.append((next_waypoint, road_option))
            # 换道成功，恢复follow状态
            self.lane_change_flag = RoadOption.LANEFOLLOW
            self.change_times += 1
        else:
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoint = last_waypoint.get_right_lane().next(self._sampling_radius)[0]
            road_option = RoadOption.CHANGELANELEFT
            self._waypoints_queue.append((next_waypoint, road_option))
            self.lane_change_flag = RoadOption.LANEFOLLOW
            self.change_times += 1

        misc.draw_waypoints(self._world, [next_waypoint])


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n%360.0

    c = current_waypoint.transform.rotation.yaw
    c = c%360.0

    diff_angle = (n-c)%180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
