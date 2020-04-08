import carla
import misc
import controller
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
D_LATERAL_RANGE = 4.5
DELTA_FI_RANGE = 60


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
        init_location = self._vehicle.get_location()
        self._current_waypoint = self._map.get_waypoint(init_location)
        self._sampling_radius = TARGET_SPEED/3.6
        self._min_distance = self._sampling_radius*MIN_DISTANCE_PERCENTAGE
        self._waypoint_resolution = 1  # distance between waypoints
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=300)
        self._buffer_size = 1
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._waypoint_resolution)[0], RoadOption.LANEFOLLOW))
        self._target_road_option = None
        # self.lane_change_flag = random.choice([RoadOption.CHANGELANELEFT,RoadOption.CHANGELANERIGHT])
        self.lane_change_flag = RoadOption.CHANGELANELEFT
        # self.lane_change_flag = RoadOption.LANEFOLLOW
        self.lane_change_start = None
        self.lane_change_duration = None

        self.change_times = 0
        self.target_waypoint = self._current_waypoint

        # self._pid_controller = controller.VehiclePIDController(vehicle,
        #                                                        args_lateral=args_lateral_dict,
        #                                                        args_longitudinal=args_longitudinal_dict)
        self._pid_controller = controller.PIDLongitudinalController(vehicle, **args_longitudinal_dict)
        self._pid_timeperiod = pid_time_period

        self.x = None
        self.y = None
        self.yaw = None
        self.a_lateral_past = 0
        self.a_lateral = 0
        self.jerk = None
        self.d_lateral = None
        self.volocity = None
        self.delta_fi = None

        self._control = carla.VehicleControl()
        self._control.steer = 0.0
        self._control.throttle = 0.0
        self._control.brake = 1.0
        self._control.hand_brake = False
        self._control.manual_gear_shift = False

        self.data = None

        self.trajectory_rotation_angle = math.radians(self._current_waypoint.transform.rotation.yaw)
        # fill waypoint trajectory queue
        self._make_waypoints_queue(self.lane_change_flag)

    def run_step(self):  # 为了得到target_waypoint以及纵向pid自动控制的throttle值
        finish = False
        if len(self._waypoints_queue) == 0:
            # self._control.steer = 0.0
            # self._control.throttle = 0.0
            # self._control.brake = 1.0
            # self._control.hand_brake = False
            # self._control.manual_gear_shift = False
            # self._vehicle.apply_control(self._control)
            finish = True
            # return self._control, finish

        self._update_buffer()

        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # target waypoint
        self.target_waypoint, self._target_road_option = self._waypoint_buffer[0]
        # move using PID controllers
        self._control.throttle = self._pid_controller.run_step(TARGET_SPEED+BIAS)
        self._control.brake = 0

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

        return self._control, finish  # 这里只控制throttle纵向控制

    def get_current_data(self, simulation_time):
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

        self.delta_fi = (self.yaw-self.target_waypoint.transform.rotation.yaw)%360
        if self.delta_fi > 180:
            self.delta_fi = self.delta_fi-360

        # print('x= %f'%self.x, 'y= %f'%self.y,
        #       'yaw= %f'%self.yaw, 'a_lateral= %f'%self.a_lateral,
        #       'jerk=%f'%self.jerk, 'simulation_time=%f'%simulation_time,
        #       'd_lateral= %f'%self.d_lateral, 'v= %f'%self.volocity)

        if self.lane_change_start is None and abs(self.d_lateral) > 2.5:
            self.lane_change_start = simulation_time
        if self.lane_change_start is not None and self.lane_change_duration is None:
            if abs(self.d_lateral) < 0.25:
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

        return np.array((self.d_lateral/D_LATERAL_RANGE,
                         self.delta_fi/DELTA_FI_RANGE))

    def _make_waypoints_queue(self, lane_change_flag):
        last_waypoint = self._current_waypoint
        for _ in range(45):
            next_waypoint = list(last_waypoint.next(self._waypoint_resolution))[0]
            road_option = RoadOption.LANEFOLLOW
            self._waypoints_queue.append((next_waypoint, road_option))
            last_waypoint = self._waypoints_queue[-1][0]

        if lane_change_flag == RoadOption.CHANGELANELEFT:
            next_waypoint = last_waypoint.get_left_lane().next(self._waypoint_resolution)[0]
            self._waypoints_queue.append((next_waypoint, lane_change_flag))
            self.change_times += 1
        elif lane_change_flag == RoadOption.CHANGELANERIGHT:
            next_waypoint = last_waypoint.get_right_lane().next(self._waypoint_resolution)[0]
            self._waypoints_queue.append((next_waypoint, lane_change_flag))
            self.change_times += 1
        last_waypoint = self._waypoints_queue[-1][0]

        for _ in range(104):
            next_waypoint = list(last_waypoint.next(self._waypoint_resolution))[0]
            road_option = RoadOption.LANEFOLLOW
            self._waypoints_queue.append((next_waypoint, road_option))
            last_waypoint = self._waypoints_queue[-1][0]

    def _update_buffer(self):
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break


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
