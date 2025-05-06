
#!/usr/bin/python3
from names_and_constants import SIMULATOR_FLAG, SHOW_IMGS, RANDOM_START, EVENT_SETTINGS, EVENT_CONFIGS # deleted completely the speed challenge
import sys
import os
import numpy as np
import cv2 as cv
from time import time, sleep
from numpy.linalg import norm
from collections import deque
import names_and_constants as nac
from extra.giveme_fruits import compute_optimal_path

if not SIMULATOR_FLAG:
    from automobile_data_interface import Automobile_Data
else:
    from automobile_data_interface import Automobile_Data
    
from path_planning4_mod import PathPlanning
from controller3 import Controller
from controllerSP import ControllerSpeed
from controllerAG import ControllerSpeed as ControllerBL
from detection import Detection
from environmental_data_simulator import EnvironmentalData
from obstacle2 import Obstacle
import helper_functions as hf

from parkman import Maneuvers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SELECTED_EVENT = "test2" # "tunnel", "round","no_lane_left/right", "highway", "crosswalk", "parking" , "test"

# Based on the path given for the arena challenge
END_NODE_ARENA = 149
ARENA = False # TODO: Move this flag to names and constant and implement args for it --arena
USE_FRUITS_GENERATED_PATH = False
# RANDOM START
if RANDOM_START:
    # USED FOR SPECIFIC PATHS DURING TESTING
    if SELECTED_EVENT in EVENT_CONFIGS:
        config = EVENT_CONFIGS[SELECTED_EVENT]
        STARTING_COORDS = config["starting_coords"]  
        CHECKPOINTS = config["checkpoints"]
        #END_NODE = EVENT_CONFIGS[SELECTED_EVENT]["checkpoints"][-1]
        END_NODE = CHECKPOINTS[-1]
        print(f"Starting coords: {STARTING_COORDS}, Checkpoints: {CHECKPOINTS}, End node: {END_NODE}")
        OVERTAKE_COUNTER = [25]                 # TODO: IMPLEMENT BETTER OVERTAKE CONDITION
        GPS_FOR_START_ONLY = False
        USE_FRUITS_GENERATED_PATH = False #we are not using fruits path for when we are testing specific events
    # MANUALLY WRITE THE PATH     
    elif ARENA: 
        STARTING_COORDS = [0.00, 0.00]  # IS GIVEN
        CHECKPOINTS = [455, 465]        # IS GIVEN
        OVERTAKE_COUNTER = [25]
        GPS_FOR_START_ONLY = False
    # GET THE BEST "FRUITS" PATH FROM RANDOM POSITION     
    else: 
        STARTING_COORDS = [17.27, 5.6] # GET FROM GPS 
        CHECKPOINTS = compute_optimal_path(start_node=472) # GET FROM FRUITS
        END_NODE = CHECKPOINTS[-1]  #get the last node from the path
        OVERTAKE_COUNTER = [25]
        GPS_FOR_START_ONLY = True
# DEFAULT START
else:
    STARTING_COORDS = [-42, -42]                            # DEFAULT START POSITION
    CHECKPOINTS = [322,149,123,450, 444, 96]  #, 120, 102, 130, 172, 180, 420, 352, 502, 160]                   472, 322, 149, 123, 
    END_NODE = CHECKPOINTS[-1]
    OVERTAKE_COUNTER = [3, 4, 23]
    GPS_FOR_START_ONLY = False

# STARTING_COORDS = [3.17, 2.55]    # SEMAPHORS ENTER DOWN
# STARTING_COORDS = [2.77, 5.82]    # SEMAPHORS ENTER UP
# STARTING_COORDS = [1.75, 3.97]    # SEMAPHORS ENTER LEFT
# STARTING_COORDS = [4.25, 4.39]    # SEMAPHORS ENTER RIGHT

# STARTING_COORDS = [5.19, 13.0]    # HIGHWAY ENTRANCE LEFTcing up)
# STARTING_COORDS = [9.62, 4.38]    # BUS LANE AFTER CROSS
# STARTING_COORDS = [17.49, 9.81]    # ROUNDABOUT EXIT DOWN
# STARTING_COORDS = [4.11, 0.76]    # DOWN PART MIDLE CITY

ALWAYS_USE_VISION_FOR_STOPLINES = True


ALWAYS_TRUST_GPS = False    # if true the car will always trust the gps (bypass)
ALWAYS_DISTRUST_GPS = True # if true, the car will always distrust the gps (bypass)   change to false if start with imu
assert not (ALWAYS_TRUST_GPS and ALWAYS_DISTRUST_GPS), 'ALWAYS_TRUST_GPS and ALWAYS_DISTRUST_GPS cannot be both True'
# SP32_CAM
ALWAYS_TRUST_ESP32 = False    # if true the car will always trust the ESP32 CAMERA CLASSIFICATION

# Templates for obstacle detection
num_tem = 5
tem = []
tem.append(cv.imread("models/templates/car1.png"))
tem.append(cv.imread("models/templates/car2.png"))
tem.append(cv.imread("models/templates/car3.png"))
tem.append(cv.imread("models/templates/car4.png"))
tem.append(cv.imread("models/templates/car5.png"))
# tem.append(cv.imread("templates/rb1.png"))
# tem.append(cv.imread("templates/rb2.png"))
# tem.append(cv.imread("templates/rb3.png"))
# tem.append(cv.imread("templates/rb4.png"))
# tem.append(cv.imread("templates/rb5.png"))
obs = Obstacle(tem, num_tem)

park = Maneuvers()


class State():
    def __init__(self, name=None, method=None, activated=False):
        self.name = name
        self.method = method
        self.active = activated
        self.start_time = None
        self.start_position = None
        self.start_distance = None
        self.just_switched = False
        self.interrupted = False
        # variables specific to state, can be freely assigned
        self.var1 = None
        self.var2 = None
        self.var3 = None
        self.var4 = None

    def __str__(self):
        return self.name.upper() if self.name is not None else 'None'

    def run(self):
        self.method()


ALWAYS_ON_ROUTINES = [nac.UPDATE_STATE, nac.CONTROL_FOR_SIGNS]


class Routine():
    def __init__(self, name, method, activated=False):
        self.name = name
        self.method = method
        self.active = activated
        self.start_time = None
        self.start_position = None
        self.start_distance = None
        self.var1 = None
        self.var2 = None
        self.var3 = None

    def __str__(self):
        return self.name

    def run(self):
        self.method()


EVENT_TYPES = [nac.INTERSECTION_STOP_EVENT,             #0
               nac.INTERSECTION_TRAFFIC_LIGHT_EVENT,    #1    
               nac.INTERSECTION_PRIORITY_EVENT,         #2
               nac.ROUNDABOUT_EVENT,                    #3
               nac.CROSSWALK_EVENT,                     #4
               nac.PARKING_EVENT,                       #5
               nac.HIGHWAY_EXIT_EVENT,                  #6
               nac.HIGHWAY_ENTRANCE_EVENT,              #7
               nac.TUNNEL_EVENT,                        #8
               nac.NO_LANE_EVENT]                       #9


class Event:
    def __init__(self, name=None, dist=None, point=None, yaw_stopline=None,
                 path_ahead=None, length_path_ahead=None, curvature=None):
        self.name = name  # name/type of the event
        self.dist = dist  # distance of event from start of path
        # [x,y] position on the map of the event
        self.point = point
        self.yaw_stopline = yaw_stopline  # yaw of the stop line at the event
        # <++>
        # if self.yaw_stopline is None:
        #     self.yaw_stopline = 0.0
        # sequence of points after the event,
        # only for intersections or roundabouts
        self.path_ahead = path_ahead
        # length of the path after the event,
        # only for intersections or roundabouts
        self.length_path_ahead = length_path_ahead
        self.curvature = curvature  # curvature of the path ahead of the event

    def __str__(self):
        return self.name.upper() if self.name is not None else 'None'


CONDITIONS = {
        # if true, the car is in presence of a
        # dotted line and is allowed to overtake it
        nac.CAN_OVERTAKE: False,
        # if true, the car is in a highway, the speed
        # should be higher on highway,
        nac.HIGHWAY:      False,
        # and the position is too far from the path it will be set to false
        nac.CAR_ON_PATH:  True,
        # if true, the car is rerouting, for example at the
        # beginning or after a roadblock
        nac.REROUTING:    True,
        # if true, the car is in the tunnel
        nac.TUNNEL:       False
}

ACHIEVEMENTS = {
        nac.PARK_ACHIEVED:    False
}

# ==============================================================
# ========================= PARAMTERS ==========================
# ==============================================================
# signs
SIGN_DIST_THRESHOLD = 0.5
SIGN_CLASSIFY_THRESHOLD = 0.8

# sempahores
#SEMAPHORE_IS_ALWAYS_GREEN = False if not SIMULATOR_FLAG else True
SEMAPHORE_IS_ALWAYS_GREEN = True

DEQUE_OF_PAST_FRAMES_LENGTH = 50
DISTANCES_BETWEEN_FRAMES = 0.03

# Yaw
APPLY_YAW_CORRECTION = False
ENCODER_POS_FREQ = 100.0             # [Hz] frequency of encoder position messages

# Vehicle driving parameters
MIN_SPEED = -0.3                    # [m/s]     minimum speed
MAX_SPEED = 2.5                     # [m/s]     maximum speed
MAX_ACCEL = 5.5                     # [m/ss]    maximum accel
MAX_STEER = 28.0                    # [deg]     maximum steering angle

# Vehicle parameters
LENGTH = 0.45                     # [m]  car body length
WIDTH = 0.18                       # [m]  car body width
BACKTOWHEEL = 0.10                  # [m]  distance of the wheel and the car body
WHEEL_LEN = 0.03

# STOPLINES
STOPLINE_APPROACH_DISTANCE = 0.4   #0.4
STOPLINE_STOP_DISTANCE = 0.15 # 0.15 simulation       # 0.1 #in the true map
assert STOPLINE_STOP_DISTANCE <= STOPLINE_APPROACH_DISTANCE

# <++>
# STOP_WAIT_TIME = 0.001*3.0  
STOP_WAIT_TIME = 3.0 
# local tracking
OPEN_LOOP_PERCENTAGE_OF_PATH_AHEAD = 0.6  # 0.6
# distance from previous stopline from which is possible to start detecting a stop line again
STOPLINE_DISTANCE_THRESHOLD = 0.2 #0.2
POINT_AHEAD_DISTANCE_LOCAL_TRACKING = 0.3  # 0.3

# speed control
# multiplier for desired speed, used to regulate highway speed
ACCELERATION_CONST = 1.2
SLOW_DOWN_CONST = 0.3

# highway exit
# [m] go straight for this distance in orther to exit the hihgway
STRAIGHT_DIST_TO_EXIT_HIGHWAY = 0.8

# Rerouting
# distance between 2 consecutive measure of the gps for
# the kalmann filter to be considered converged
GPS_DISTANCE_THRESHOLD_FOR_CONVERGENCE = 0.2
GPS_SAMPLE_TIME = 0.25  # [s] time between 2 consecutive gps measurements
GPS_CONVERGENCE_PATIANCE = 0  # 2 #iterations to consider the gps converged
GPS_TIMEOUT = 5.0  # [s] time to wait to have gps signal


# end state
# [m] distance from the end of the path for the car
# to be considered at the end of the path
END_STATE_DISTANCE_THRESHOLD = 0.3

# PARKING
PARKING_DISTANCE_SLOW_DOWN_THRESHOLD = 0.7  # 1.0
PARKING_DISTANCE_STOP_THRESHOLD = 0.1       # 0.1
SUBPATH_LENGTH_FOR_PARKING = 300            # length in samples of the path to consider around the parking position, max
MAX_PARK_SEARCH_DIST = 2.0                  # [m] max distance to search for parking
MIN_PARK_SEARCH_DIST = 0.7                  # [m] min distance to search for parking
IDX_OFFSET_FROM_SAVED_PARK_POSITION = 150   # index offset from the saved parking position; value of 150 in 2024
PARK_SIGN_DETETCTION_PATIENCE = 8.0         # [s] max seconds to wait for a sign to be available
PARK_SEARCH_SPEED = 0.1                     # [m/s] speed to search for parking
PARK_MANOUVER_SPEED = 0.15                 # [m/s] speed to perform the parking manouver

DIST_SIGN_FIRST_S_SPOT = 0.7               # [m] distance from the sign to the first parking spot
DIST_S_SPOTS = 0.85                         # [m] distance to go forward to start parking
FURTHER_DIST_S = 0.70                       # [m] distance to proceed further in order to perform the s manouver
DIST_FORWARD = 0.7

S_ANGLE = 29.0                              # [deg] angle to perform the s manouver
DIST_2S = 0.40                              # 0.38 #[m] distance to perform the 2nd part of s manouver
DIST_4S = 0.05                              # [m] dsemaphoreistance to perform the 4th part of s manouver
STEER_ACTUATION_DELAY_PARK = 0.5            # [s] delay to perform the steering manouver
SLEEP_AFTER_STOPPING = 0.3                  # [s] WARNING: this stops the state machine. So be careful increasing it
STEER_ACTUATION_DELAY = 0.3                 # [s] delay to perform the steering manouver

# TUNNEL
TUNNEL_SPEED = 0.05                        # [m/s] speed in the tunnel


# OBSTACLES
OBSTACLE_IS_ALWAYS_PEDESTRIAN = False
OBSTACLE_IS_ALWAYS_CAR = False

# obstacle classification
MIN_DIST_BETWEEN_OBSTACLES = 0.5             # dont detect obstacle for this distance after detecting one of them
OBSTACLE_DISTANCE_THRESHOLD = 0.6            # [m] distance from the obstacle to consider it as an obstacle
OBSTACLE_CONTROL_DISTANCE = 0.3              # distance to where to stop wrt the obstacle
OBSTACLE_CLASSIFY_THRESHOLD = 0.80           # confidence level of the classifying
OBSTACLE_IMGS_CAPTURE_START_DISTANCE = 0.48  # dist from where we capture imgs
OBSTACLE_IMGS_CAPTURE_STOP_DISTANCE = 0.31   # dist up to we capture imgs
assert OBSTACLE_IMGS_CAPTURE_STOP_DISTANCE > OBSTACLE_CONTROL_DISTANCE
# pedestrian
PEDESTRIAN_CONTROL_DISTANCE = 0.5   # [m] distance to keep from the pedestrian
PEDESTRIAN_TIMEOUT = 2.0             # [s] time to w8 after the pedestrian cleared the road
# car
TAILING_DISTANCE = 0.30   # [m] distance to keep from the vehicle while tailing
# overtake static car
OVERTAKE_STEER_ANGLE = 27.0  # [deg]
OVERTAKE_STATIC_CAR_SPEED = 0.2  # [m/s]
OT_STATIC_SWITCH_1 = 0.3
OT_STATIC_SWITCH_2 = 0.55 # BFMC_2023
OT_STATIC_SWITCH_2 = 0.40 # good for simulation # BFMC_2024
OT_STATIC_LANE_FOLLOW = 0.3
# overtake moving car
OVERTAKE_MOVING_CAR_SPEED = 0.5  # [m/s]
OT_MOVING_SWITCH_1 = 0.27  # [m]
OT_MOVING_LANE_FOLLOW = 2.00  # [m]
OT_MOVING_SWITCH_2 = 0.27  # [m]


# CHECKS
# [m] max distance from the lane to trip the state checker
MAX_DIST_AWAY_FROM_LANE = 0.8
MAX_ERROR_ON_LOCAL_DIST = 0.05  # [m] max error on the local distance


# ==============================================================
# =========================== BRAIN ============================
# ==============================================================
class Brain:
    def __init__(self,
                 car: Automobile_Data,
                 controller: Controller,
                 controller_sp: ControllerSpeed,
                 controller_ag: ControllerBL,
                 env: EnvironmentalData,
                 detection: Detection,
                 path_planner: PathPlanning,
                 checkpoints = None,
                 desired_speed = 0.3,
                 debug=True):
        print("Initialize brain")
        self.car = car
        self.controller = controller
        self.controller_sp = controller_sp
        self.controller_ag = controller_ag
        self.detect = detection
        self.path_planner = path_planner
        self.env = env

        self.car.drive(speed=0.0, angle=0.0)
        self.laremilputas = None

        # navigation instruction is a list of tuples:
        self.navigation_instructions = []
        # events are an ordered list of tuples: (type , distance from start, x y position)
        self.events = []
        if checkpoints is not None:
            self.checkpoints = checkpoints
        else:
            self.checkpoints = CHECKPOINTS
        self.checkpoint_idx = 0
        self.desired_speed = desired_speed

        # pedestrian variables
        self.flag_pedestrian_in_the_way = False
        self.flag_seen_pedestrian = False
        self.pedestrian_on_the_crosswalk = False
        # current and previous states (class State)
        self.curr_state = State()
        self.prev_state = State()
        # previous and next event (class Event)
        self.prev_event = Event()
        self.next_event = Event()
        self.second_next_event = Event()
        self.event_idx = 0


        self.tunnel_integral_error = 0.0      ## <++> BFMC_2025
        self.tunnel_last_time = None          ## <++> BFMC_2025

        # stop line with higher precision
        self.stopline_distance_median = 1.0
        self.car_dist_on_path = 0  # init

        # debug
        self.debug = debug
        if self.debug and SHOW_IMGS:
            cv.namedWindow('brain_debug', cv.WINDOW_NORMAL)
            self.debug_frame = None

        self.conditions = CONDITIONS
        self.achievements = ACHIEVEMENTS

        # INITIALIZE STATES

        self.states = {
            nac.START_STATE:             State(nac.START_STATE, self.start_state),
            nac.END_STATE:               State(nac.END_STATE, self.end_state),
            # lane following, between intersections or roundabouts
            nac.LANE_FOLLOWING:          State(nac.LANE_FOLLOWING, self.lane_following),
            # intersection navigation, further divided into the possible directions [left, right, straight]
            nac.APPROACHING_STOPLINE:    State(nac.APPROACHING_STOPLINE, self.approaching_stopline),
            nac.TRACKING_LOCAL_PATH:     State(nac.TRACKING_LOCAL_PATH, self.tracking_local_path),
            # waiting states  
            nac.WAITING_FOR_GREEN:       State(nac.WAITING_FOR_GREEN, self.waiting_for_green),
            nac.WAITING_AT_STOPLINE:     State(nac.WAITING_AT_STOPLINE, self.waiting_at_stopline),
            # overtaking manouver  
            nac.OVERTAKING_STATIC_CAR:   State(nac.OVERTAKING_STATIC_CAR, self.overtaking_static_car),
            nac.OVERTAKING_MOVING_CAR:   State(nac.OVERTAKING_MOVING_CAR, self.overtaking_moving_car),
            nac.TAILING_CAR:             State(nac.TAILING_CAR, self.tailing_car),
            # parking  
            nac.PARKING:                 State(nac.PARKING, self.parking),
            # crosswalk navigation  
            nac.CROSSWALK_NAVIGATION:    State(nac.CROSSWALK_NAVIGATION, self.crosswalk_navigation),
            nac.TUNNEL_SPEED_CURVE:      State(nac.TUNNEL_SPEED_CURVE, self.tunnel_speed_curve),
            nac.NO_LANE_STATE:           State(nac.NO_LANE_STATE, self.no_lane)
        }

        # INITIALIZE ROUTINES
        self.routines = {
            nac.FOLLOW_LANE:            Routine(nac.FOLLOW_LANE,  self.follow_lane),
            nac.DETECT_STOPLINE:        Routine(nac.DETECT_STOPLINE,  self.detect_stopline),
            nac.SLOW_DOWN:              Routine(nac.SLOW_DOWN,  self.slow_down),
            nac.ACCELERATE:             Routine(nac.ACCELERATE,  self.accelerate),
            nac.CONTROL_FOR_SIGNS:      Routine(nac.CONTROL_FOR_SIGNS,  self.control_for_signs),
            nac.CONTROL_FOR_CAR:        Routine(nac.CONTROL_FOR_CAR,  self.control_for_car),
            nac.CONTROL_FOR_PEDESTRIAN: Routine(nac.CONTROL_FOR_PEDESTRIAN,  self.control_for_pedestrian),
            nac.UPDATE_STATE:           Routine(nac.UPDATE_STATE, self.update_state),
            nac.DRIVE_DESIRED_SPEED:    Routine(nac.DRIVE_DESIRED_SPEED, self.drive_desired_speed),
            nac.FOLLOW_LANE_RIGHT:      Routine(nac.FOLLOW_LANE_RIGHT,  self.follow_lane_right),
            nac.FOLLOW_LANE_LEFT:       Routine(nac.FOLLOW_LANE_LEFT,  self.follow_lane_left) 
            
        }
        self.active_routines_names = []

        # # BFMC_2023
        # self.sign_points = np.load('data/sign_points.npy')
        # self.sign_types = np.load('data/sign_types.npy').astype(int)
        # assert len(self.sign_points) == len(self.sign_types)
        # FROM MATTEO_GRANDE
        self.sign_points = np.loadtxt('data/sign_points.txt', dtype=float)
        self.sign_types = np.loadtxt('data/sign_types.txt', dtype=int)
        assert len(self.sign_points) == len(self.sign_types), f'len(self.sign_points): {len(self.sign_points)}, len(self.sign_types): {len(self.sign_types)}'
        self.sign_seen = np.zeros(len(self.sign_types))
        self.curr_sign = nac.NO_SIGN
        self.past_frames = deque(maxlen=DEQUE_OF_PAST_FRAMES_LENGTH)

        self.frame_for_stopline_angle = None
        self.last_run_call = time()

        self.start_node_validated = False

        self.stopline_counter = 0
        self.pedestrian_type = None

        print('Brain initialized')
        if not RANDOM_START:
            print('Waiting for start semaphore...')
            sleep(3.0)
            while True:
                semaphore_start_state = self.env.get_semaphore_state(nac.START)
                if SEMAPHORE_IS_ALWAYS_GREEN:
                    semaphore_start_state = nac.GREEN
                if semaphore_start_state == nac.GREEN:
                    break
        sleep(0.1)

        # <++>
        # <++>
        # <++>
        # <++>
        start_time = time()
        while True:
            # get closest node
            if not ALWAYS_DISTRUST_GPS or GPS_FOR_START_ONLY:
                curr_time = time()
                #curr_pos = np.array([self.car.x_est, self.car.y_est])
                curr_pos = np.array(STARTING_COORDS) 
                closest_node, distance = self.path_planner.get_closest_node_start(curr_pos, self.car.yaw)
                self.car.publish_closest_node(float(closest_node))  ##
                sleep(3.0)
                if len(self.car.x_buffer) >= 5:
                    print(f'Waiting for gps: {(curr_time- start_time):.1f}/{GPS_TIMEOUT}')
                    self.checkpoints[self.checkpoint_idx] = closest_node
                    if USE_FRUITS_GENERATED_PATH:
                        checkpoints = compute_optimal_path(start_node=closest_node)
                        self.checkpoints = checkpoints
                    if distance > 5.0:
                        self.error('ERROR: REROUTING: GPS converged, but distance is too large , we are too far from the lane')
                    break
            
                if curr_time - start_time > GPS_TIMEOUT:
                    print('WARNING: ROUTE_GENERATION: No gps signal, Starting from the first checkpoint')
                    sleep(3.0)
                    break
            else:
                if STARTING_COORDS != [-42, -42]:        
                    curr_pos = np.array(STARTING_COORDS)
                    closest_node, distance = self.path_planner.get_closest_node(curr_pos)
                    self.car.publish_closest_node(float(closest_node))     ##
                    self.checkpoints[self.checkpoint_idx] = closest_node
                    if USE_FRUITS_GENERATED_PATH:
                        checkpoints = compute_optimal_path(start_node=closest_node)
                        self.checkpoints = checkpoints
                    self.car.x_est = curr_pos[0]
                    self.car.y_est = curr_pos[1]
                    print(closest_node)
                    # raise KeyboardInterrupt
                elif len(self.car.x_buffer) < 5:
                    node_coords = self.path_planner.get_coord(str(self.checkpoints[0]))
                    self.car.x_est = node_coords[0]
                    self.car.y_est = node_coords[1]
                break
            # self.car.update_estimated_state()  # <++>

     
        # if bool(self.checkpoints[self.checkpoint_idx] in
        #         self.path_planner.intersection_in or
        #         self.checkpoints[self.checkpoint_idx] in
        #         self.path_planner.ra_enter) ^ bool(self.detect.detect_stopline(
        #         self.car.frame, show_ROI=SHOW_IMGS)[0] < 0.05):
        #     # self.checkpoints[self.checkpoint_idx] = hf.switch_lane_check(
        #     #                                            closest_node, angle)
        #     print("<++>")

        # if self.checkpoints[0] in RB_NODES_RIGHT_INT:
        #     self.checkpoints.insert(1, 146)                     #BFMC_2023
        # elif self.checkpoints[0] in RB_NODES_LEFT_INT:
        #     self.checkpoints.insert(1, 145)                     #BFMC_2023

        #self.switch_to_state(nac.START_STATE)
        self.switch_to_state(nac.START_STATE)

    # =============== STATES =============== #
    def start_state(self):
        if self.curr_state.just_switched:
            self.conditions[nac.REROUTING] = True
            # Check in name and constants the comment. Maybe it is not needed anymore if full path generation can be created
            if nac.DONT_STOP_AT_NO_LANE_EVENT:
                nac.DONT_STOP_AT_NO_LANE_EVENT = False # reset the flag 
            else:
                self.car.drive_distance(0.0)  # stop

            self.curr_state.var2 = time()
            self.curr_state.just_switched = False

        # localize the car and go to the first checkpoint

        print('Generating route...')
        print(self.checkpoints)
        # get start and end nodes from the chekpoint list
        assert len(self.checkpoints) >= 2, 'List of checkpoints needs 2 or more nodes'

        # Compute path through all remaining checkpoints

        start_node = self.checkpoints[self.checkpoint_idx]
        end_node = self.checkpoints[self.checkpoint_idx+1]

        self.path_planner.compute_shortest_path(start_node, end_node)



        events = self.path_planner.augment_path(draw=SHOW_IMGS)
        # add the events to the list of events, increasing it
        self.path_planner.draw_path()
        #cv.waitKey(0) COMMENTED for debugging

        self.events = self.create_sequence_of_events(events)
        self.event_idx = 1
        self.next_event = self.events[0]
        if not self.event_idx == len(self.events):
            self.second_next_event = self.events[1]
        else:
            self.second_next_event = None
        self.prev_event.dist = 0.0
        self.car.reset_rel_pose()
        
        #print(f' ======  Events ====== ')
        #for e in self.events:
        #    print(f"{e}")
        #print(f' ======  ****** ====== ')

        # draw the path
        self.path_planner.draw_path()
        #cv.waitKey(0)
        print('Starting...')
        if self.next_event.name == nac.PARKING_EVENT:
            print('Skipping parking if its the first event')
            self.go_to_next_event()
        self.conditions[nac.REROUTING] = False
        # reset the signs seen
        self.sign_seen = np.zeros_like(self.sign_seen)

        self.car_dist_on_path = 0

        self.switch_to_state(nac.LANE_FOLLOWING)

    def end_state(self):
        self.activate_routines([nac.SLOW_DOWN])
        self.go_to_next_event()
        # start routing for next checkpoint
        self.next_checkpoint()
        self.switch_to_state(nac.START_STATE)


    def lane_following(self):  # LANE FOLLOWING ##############################
        # highway conditions
        print('in the function')
        if self.conditions[nac.HIGHWAY]:
            self.activate_routines([nac.FOLLOW_LANE,
                                    nac.DETECT_STOPLINE,
                                    nac.CONTROL_FOR_CAR,
                                    nac.ACCELERATE])
        else:
            print('entering the if')
            self.activate_routines([nac.FOLLOW_LANE,
                                    nac.DETECT_STOPLINE,
                                    nac.CONTROL_FOR_CAR,
                                    nac.DRIVE_DESIRED_SPEED])

        # check parking
        if self.next_event.name == nac.PARKING_EVENT:
            # we dont need stoplines in parking
            self.activate_routines([nac.FOLLOW_LANE,
                                    nac.CONTROL_FOR_CAR,
                                    nac.DRIVE_DESIRED_SPEED])
            self.lane_following_to_parking()

                
        elif (self.next_event.name == nac.NO_LANE_EVENT):   
            self.switch_to_state(nac.NO_LANE_STATE)
        
        # check highway entrance case
        elif self.next_event.name == nac.HIGHWAY_ENTRANCE_EVENT:
            self.lane_following_highway_entrance()

        #TUNNEL NEW 
        # if next next event is TUNNEL_EVENT switch to TUNNEL state (we use next next because the intersection stop event is not trigered as the croswalk is too close to the entrance)
        elif getattr(self.second_next_event, 'name', None) == nac.TUNNEL_EVENT:   #safe against None values
            min_distance_lidar_right = hf.get_min_distance_in_range(self.car.lidar_angles,self.car.lidar_ranges, -95, -85) 
            self.activate_routines([nac.FOLLOW_LANE,
                                    nac.DETECT_STOPLINE,
                                    nac.CONTROL_FOR_CAR,
                                    nac.DRIVE_DESIRED_SPEED])
            if (min_distance_lidar_right <= 0.4):    
                self.switch_to_state(nac.TUNNEL_SPEED_CURVE)
                self.go_to_next_event()  

        elif self.next_event.name == nac.TUNNEL_EVENT:
            min_distance_lidar_right = hf.get_min_distance_in_range(self.car.lidar_angles,self.car.lidar_ranges, -95, -85) 
            if (min_distance_lidar_right <= 0.4):    
                self.switch_to_state(nac.TUNNEL_SPEED_CURVE)

          

        # check highway exit case
        # elif self.next_event.name == nac.HIGHWAY_EXIT_EVENT:    # BFMC_2023
        #     self.lane_following_to_highway_exit()

        # end of current route, go to end state
        elif self.next_event.name == nac.END_EVENT:

           

            if self.checkpoint_idx == len(self.checkpoints) - 1:
                self.lane_following_to_end()
            else:
                self.go_to_next_event()
                # start routing for next checkpoint
                self.next_checkpoint()
                self.switch_to_state(nac.START_STATE)

        # we are approaching a stopline, check only if we are far enough from the previous stopline
        else:
            far_enough_from_prev_stopline = (self.event_idx == 1) or (self.car.dist_loc > STOPLINE_DISTANCE_THRESHOLD)
            if self.prev_event.name is not None:
                print(f'stop enough: {self.car.dist_loc}')
            if self.detect.est_dist_to_stopline < STOPLINE_APPROACH_DISTANCE and far_enough_from_prev_stopline and self.routines[nac.DETECT_STOPLINE].active:
                self.switch_to_state(nac.APPROACHING_STOPLINE)

    def lane_following_highway_entrance(self):
        '''
        Lane following until inside highway then change to right lane
        '''
        print("Highway entrance phase")
        if self.conditions[nac.HIGHWAY]:
            self.activate_routines([])
            # if self.car.filtered_left_sonar_distance <= 0.5: #checking distance from the highway separator
            if self.curr_state.just_switched:
                self.car.drive_angle(OVERTAKE_STEER_ANGLE)
                self.curr_state.var4 = self.car.encoder_distance           
                self.car.drive_speed(OVERTAKE_MOVING_CAR_SPEED)
                self.curr_state.just_switched = False
            dist = self.car.encoder_distance - self.curr_state.var4  
            assert dist > -0.05
            print(f'Switching to the right lane: {dist:.2f}/{OT_MOVING_SWITCH_2:.2f}')
            if dist > OT_MOVING_SWITCH_2:
                self.switch_to_state(nac.LANE_FOLLOWING)
                self.go_to_next_event()


    # Event: Parking <++>
    def lane_following_to_parking(self):
        dist_between_events = self.next_event.dist - self.prev_event.dist
        # Relative positioning is reset at every stopline, so we
        # can use that to approximately calculate the distance
        # to the parking spot
        approx_dist_from_parking = dist_between_events - self.car.dist_loc
        print(f'Approx dist from parking: {approx_dist_from_parking}')
        # we are reasonably close to the parking spot
        if approx_dist_from_parking < PARKING_DISTANCE_SLOW_DOWN_THRESHOLD:
            self.car.drive_speed(0.0)
            sleep(SLEEP_AFTER_STOPPING)
            self.switch_to_state(nac.PARKING)

    # # Event: Highway exit <++>
    # def lane_following_to_highway_exit(self):
    #     if self.curr_state.just_switched:
    #         self.curr_state.var1 = self.car.encoder_distance   
    #         self.curr_state.just_switched = False
    #         diff = self.car.encoder_distance - self.curr_state.var1    
    #         print("##########################################")
    #         print("diff = ", diff)
    #         print("##########################################")
    #         if diff < 2.0:
    #             print(f'Driving toward highway exit: dist so far {diff:.2f} [m]')
    #         elif diff < 3.4:
    #             self.activate_routines([])
    #             print(f'Driving toward highway exit: dist so far {diff:.2f} [m]')
    #             self.car.drive_angle(angle=0.0)
    #         else:
    #             print('Arrived at highway exit, switching to going straight for exiting')
    #             self.switch_to_state(nac.END_STATE)

    # Event: End
    def lane_following_to_end(self):
        print('Driving toward end...')
        if self.curr_state.just_switched:
            self.curr_state.var1 = self.car.encoder_distance         
            self.curr_state.just_switched = False

        self.activate_routines([nac.FOLLOW_LANE,
                                nac.CONTROL_FOR_CAR,
                                nac.DRIVE_DESIRED_SPEED])
        # NOTE End is implemented only with gps now, much more robust,
        # but cannot do it without it
        diff = self.car.encoder_distance - self.curr_state.var1          
        dist_to_end = self.next_event.dist - diff
        # dist_to_end = len(self.path_planner.path)*0.01 - diff
        print('DIST TO END: ', dist_to_end)
        if dist_to_end > END_STATE_DISTANCE_THRESHOLD:
            print(f'Driving toward end: exiting in {dist_to_end:.2f} [m]')
        elif dist_to_end > -END_STATE_DISTANCE_THRESHOLD:
            print('Arrived at end, switching to end state')
            self.switch_to_state(nac.END_STATE)
        else:
            self.error('ERROR: LANE FOLLOWING: Missed end')

    def approaching_stopline(self):
        # FOLLOW_LANE, SLOW_DOWN, DETECT_STOPLINE, CONTROL_FOR_CAR
        self.activate_routines([nac.FOLLOW_LANE,
                                nac.CONTROL_FOR_PEDESTRIAN,
                                nac.SLOW_DOWN,
                                nac.DETECT_STOPLINE])

        if self.curr_state.just_switched:
            cv.imwrite(f'asl/asl_{int(time() * 1000)}.png', self.car.frame)
            self.curr_state.just_switched = False

        decide_next_state = self.approaching_stopline_vision()

        if decide_next_state:
            print('Deciding next state, based on next event...')
            # print(f'debug: {self.checkpoint_idx}')
            if not nac.RANDOM_START or nac.RANDOM_START :
                self.stopline_counter += 1
            elif EVENT_SETTINGS:
                self.stopline_counter += 1
            elif self.checkpoint_idx:
                self.stopline_counter += 1
            # print(self.stopline_counter)
            self.conditions[nac.HIGHWAY] = False
            #self.conditions[nac.NO_LANE] = False  # <++> BFMC_2025
            next_event_name = self.next_event.name
            # Events with stopline
            if next_event_name == nac.INTERSECTION_STOP_EVENT:      
                self.switch_to_state(nac.WAITING_AT_STOPLINE)
            elif next_event_name == nac.INTERSECTION_TRAFFIC_LIGHT_EVENT:
                self.switch_to_state(nac.WAITING_FOR_GREEN)
            elif next_event_name == nac.INTERSECTION_PRIORITY_EVENT:
                self.switch_to_state(nac.TRACKING_LOCAL_PATH)
            elif next_event_name == nac.ROUNDABOUT_EVENT:
                self.switch_to_state(nac.TRACKING_LOCAL_PATH)
                self.switch_to_state(nac.TRACKING_LOCAL_PATH)
            elif next_event_name == nac.CROSSWALK_EVENT:
                # directly go to lane keeping, the pedestrian will
                # be managed in that state
                self.switch_to_state(nac.CROSSWALK_NAVIGATION)
            # Events without stopline = LOGIC ERROR
            elif next_event_name == nac.PARKING_EVENT:
                self.error('WARNING: UNEXPECTED STOP LINE FOUND WITH PARKING AS NEXT EVENT')
            elif next_event_name == nac.HIGHWAY_ENTRANCE_EVENT:
                self.go_to_next_event()
                self.error('WARNING: UNEXPECTED STOP LINE FOUND WITH HIGHWAY ENTRANCE AS NEXT EVENT')
                print('Going to the next event')
            elif next_event_name == nac.HIGHWAY_EXIT_EVENT:
                self.error('WARNING: UNEXPECTED STOP LINE FOUND WITH HIGHWAY EXIT AS NEXT EVENT')
            #elif next_event_name == nac.END_EVENT and self.checkpoints[-1] == END_NODE: # BFMC_2025 02MAY -- this way we end the runs only at stoplines instead of at the random in the middle of the road
            #    self.switch_to_state(nac.END_STATE)
            else:
                self.error('ERROR: UNEXPECTED STOP LINE FOUND WITH UNKNOWN EVENT AS NEXT EVENT')
            self.activate_routines([])  # deactivate all routines

    
    # Substate
    def approaching_stopline_vision(self):
        dist = self.detect.est_dist_to_stopline
        # #check if we are here by mistake
        print(f'debug: est_dist_to_stopline: {dist}')
        if dist > STOPLINE_APPROACH_DISTANCE:
            self.switch_to_state(nac.LANE_FOLLOWING)
            return False
        # we have a median => we have an accurate position for the stopline
        if self.stopline_distance_median is not None:
            print('Driving towards stop line... at distance: ',
                  self.stopline_distance_median)
            self.activate_routines([nac.FOLLOW_LANE,
                                    nac.SLOW_DOWN,
                                    nac.CONTROL_FOR_PEDESTRIAN])  # SLOW_DOWN
            dist_to_drive = self.stopline_distance_median - self.car.encoder_distance
            self.car.drive_distance(dist_to_drive)
            if dist_to_drive < STOPLINE_STOP_DISTANCE:
                print(f'Arrievd at stop line. Using median distance: {self.stopline_distance_median}')
                print(f'                           encoder distance: {self.car.encoder_distance:.2f}')
                # sleep(1.0)
                decide_next_state = True
            else:
                decide_next_state = False
        # alternative, if we don't have a median, we just use the
        # (possibly inaccurate) network estimaiton
        else:
            print('WARNING: APPROACHING_STOPLINE: stop distance may be imprecise')
            if dist < STOPLINE_STOP_DISTANCE:
                print('Stopped at stop line. Using network distance: ', self.detect.est_dist_to_stopline)
                decide_next_state = True
                dist_from_line = dist
                assert dist_from_line < 0.5, f'dist_from_line is too large, {dist_from_line:.2f}'
            else:
                decide_next_state = False
        return decide_next_state


    def tracking_local_path(self):
        # var1=local_path_cf, 
        # var2=distance travelled
        # var3=None 
        # var4=intersection direction / ra pred avg
        print('State: tracking_local_path')
        # self.activate_routines([nac.DRIVE_DESIRED_SPEED])
        self.activate_routines([])
        if self.curr_state.just_switched:
            stopline_position = self.next_event.point
            stopline_yaw = self.next_event.yaw_stopline
            # local path in the stop line frame
            local_path_slf_rot = self.next_event.path_ahead

            _, stopline_y, _ = self.detect.detect_stopline(self.car.frame, show_ROI=SHOW_IMGS)
            e2 = stopline_y
            if self.stopline_distance_median is not None:
                print('We HAVE the median, using median estimation')
                print(len(self.routines[nac.DETECT_STOPLINE].var2))
                d = self.stopline_distance_median - self.car.encoder_distance
            # we do not have an accurate position for the stopline
            else:
                print('We DONT have the median, using simple net estimation')
                print(len(self.routines[nac.DETECT_STOPLINE].var2))
                if self.detect.est_dist_to_stopline < STOPLINE_APPROACH_DISTANCE:
                    d = self.detect.est_dist_to_stopline
                else:
                    d = 0.0
            car_position_slf = -np.array([+d+0.33, +e2])

            # get orientation of the car in the stop line frame
            yaw_car = self.car.yaw
            yaw_mult_90 = hf.get_yaw_closest_axis(yaw_car)
            # get the difference from the closest multiple of 90deg
            alpha = hf.diff_angle(yaw_car, yaw_mult_90)
            # alpha_true = alpha
            print(f'alpha true: {np.rad2deg(alpha):.1f}')
            alpha = self.detect.detect_yaw_stopline(self.car.frame, SHOW_IMGS and False) * 0.8
            print(f'alpha est: {np.rad2deg(alpha):.1f}')
            # assert abs(alpha) < np.pi/6, f'Car orientation wrt stopline is too big, it needs to be better aligned, alpha = {alpha}'

            # get position of the car in the stop line frame
            # NOTE: rotation first if we ignore the laterale error and consider only the euclidean distance from the line
            # cf = car frame
            local_path_cf = (local_path_slf_rot @ hf.rot_matrix(alpha)) - car_position_slf
            # rotate from slf to cf
            self.curr_state.var1 = local_path_cf
            # Every time we stop for a stopline, we reset the local frame of reference
            self.car.reset_rel_pose()
            self.curr_state.just_switched = False
            if self.next_event.name.startswith('intersection'):
                hf.determine_intersection_direction(self, local_path_cf)
            else:
                self.curr_state.var4 = 0

            hf.show_local_path_just_switched(self, alpha, stopline_yaw, car_position_slf, local_path_cf, stopline_position, SHOW_IMGS)

        D = POINT_AHEAD_DISTANCE_LOCAL_TRACKING
        # track the local path using simple pure pursuit
        local_path = self.curr_state.var1
        car_pos_loc = np.array([self.car.x_loc, self.car.y_loc])
        local_path_cf = local_path - car_pos_loc
        dist_path = norm(local_path_cf, axis=1)
        # get idx of car position on the path
        idx_car_on_path = np.argmin(dist_path)
        dist_path = dist_path[idx_car_on_path:]
        dist_path = np.abs(dist_path - D)
        # get idx of point ahead
        # idx_point_ahead = np.argmin(dist_path) + idx_car_on_path
        idx_point_ahead = np.round(self.car.dist_loc*100) # NEW TRYING BFMC_2024t
        print(f'idx_point_ahead: {idx_point_ahead} / {len(local_path_cf)}')
        local_path_cf = local_path_cf @ hf.rot_matrix(self.car.yaw_loc)

        hf.show_local_path(self, car_pos_loc, SHOW_IMGS)

        # print(self.curr_state.var2)
        # print(self.curr_state.var3)
        # print(self.curr_state.var4)

        # the local path is straight
        if (np.abs(hf.get_curvature(local_path_cf)) < 0.1 and not self.next_event.name.startswith("roundabout")):
            print('straight')
            # max_idx = len(local_path_cf)-60  # dont follow until the end
            # max_idx = len(local_path_cf)-1  # dont follow until the end # BFMC_2023
            max_idx = 30 # BFMC_2024
        else:  # curvy path
            max_idx = len(local_path_cf)-1  # follow until the end 
            print('curvy')


        # State exit conditions
        if idx_point_ahead >= max_idx:  # we reached the end of the path    
            self.switch_to_state(nac.LANE_FOLLOWING)
            self.go_to_next_event()
        elif self.next_event.name.startswith("intersection"):
            hf.navigate_intersection(self, SHOW_IMGS)
        elif self.next_event.name.startswith("roundabout"):
            print(f'idx (ARGMIN): {idx_point_ahead}')
            idx_point_ahead = np.round(self.car.dist_loc*100)
            if idx_point_ahead >= max_idx:  # we reached the end of the path
                self.switch_to_state(nac.LANE_FOLLOWING)
                self.go_to_next_event()
            print(f'idx (DIST_LOC): {idx_point_ahead}')
            hf.navigate_roundabout(self, idx_point_ahead, max_idx, SHOW_IMGS)
        else:  # we are still on the path
            hf.navigate_open_loop(self, local_path_cf, idx_point_ahead, idx_car_on_path, SHOW_IMGS)


    def waiting_for_green(self):
        event_p = self.next_event.point
        # event_x = self.next_event.point[0]
        # event_y = self.next_event.point[1]
        semaphore, tl_state = self.env.get_closest_semaphore_state(np.array(event_p))
        # semaphore, tl_state = self.env.get_closest_semaphore_state(np.array([event_x, event_y]))
        print(f'Waiting at semaphore: {semaphore}, state: {tl_state}')
        if self.curr_state.just_switched:
            self.activate_routines([])
            if tl_state != nac.GREEN:
                self.car.drive_speed(0.0)
            # publish traffic light
            self.env.publish_obstacle(nac.TRAFFIC_LIGHT, self.car.x_est, self.car.y_est)
            self.curr_state.just_switched = False
        if tl_state == nac.GREEN or SEMAPHORE_IS_ALWAYS_GREEN:
            self.switch_to_state(nac.TRACKING_LOCAL_PATH)
            self.switch_to_state(nac.TRACKING_LOCAL_PATH)

    def waiting_at_stopline(self):
        EXTRA_TIME = 2.0 if self.stopline_counter == 50 else 0.0 # NO NEED FOR EXTRA TIME ?!
        print(f'debug extratime: {EXTRA_TIME}')
        # no routines
        self.activate_routines([])
        if STOP_WAIT_TIME > 0.0:
            if self.curr_state.just_switched:
                self.activate_routines([])
                self.car.drive_speed(0.0)
                self.curr_state.just_switched = False
            if (time() - self.curr_state.start_time) > STOP_WAIT_TIME + EXTRA_TIME:
                self.switch_to_state(nac.TRACKING_LOCAL_PATH)
                self.switch_to_state(nac.TRACKING_LOCAL_PATH)
        else:
            self.switch_to_state(nac.TRACKING_LOCAL_PATH)
            self.switch_to_state(nac.TRACKING_LOCAL_PATH)

    def overtaking_static_car(self):
        self.activate_routines([])
        # states
        OT_SWITCHING_LANE = 1
        OT_LANE_FOLLOWING = 2
        OT_SWITCHING_BACK = 3

        if self.curr_state.just_switched:
            self.curr_state.var1 = (OT_SWITCHING_LANE, True)
            self.curr_state.var2 = self.car.encoder_distance
            self.env.publish_obstacle(nac.STATIC_CAR_ON_ROAD,
                                      self.car.x_est, self.car.y_est)
            self.curr_state.just_switched = False
        sub_state, just_sub_switched = self.curr_state.var1
        dist_prev_manouver = self.curr_state.var2
        if sub_state == OT_SWITCHING_LANE:
            if just_sub_switched:
                self.car.drive_angle(-OVERTAKE_STEER_ANGLE)
                dist_prev_manouver = self.car.encoder_distance           
                self.car.drive_speed(OVERTAKE_STATIC_CAR_SPEED)
                just_sub_switched = False
            dist = self.car.encoder_distance - dist_prev_manouver        
            assert dist > -0.05
            print(f'Switching lane: {dist:.2f}/{OT_STATIC_SWITCH_1:.2f}')
            if dist > OT_STATIC_SWITCH_1:
                sub_state, just_sub_switched = OT_LANE_FOLLOWING, True
                dist_prev_manouver = self.car.encoder_distance           
        elif sub_state == OT_LANE_FOLLOWING:
            self.activate_routines([nac.FOLLOW_LANE])
            dist = self.car.encoder_distance - dist_prev_manouver        
            print(f'Following lane: {dist:.2f}/{OT_STATIC_LANE_FOLLOW:.2f}')
            #Added double check with the sonar distance
            if (dist > OT_STATIC_LANE_FOLLOW) and (self.car.filtered_right_sonar_distance > 0.5):
                sub_state, just_sub_switched = OT_SWITCHING_BACK, True
                dist_prev_manouver = self.car.encoder_distance           
        elif sub_state == OT_SWITCHING_BACK:
            if just_sub_switched:
                self.car.drive_angle(OVERTAKE_STEER_ANGLE)
                dist_prev_manouver = self.car.encoder_distance           
                just_sub_switched = False
            dist = self.car.encoder_distance - dist_prev_manouver        
            assert dist > -0.05
            print(f'Switching back: {dist:.2f}/{OT_STATIC_SWITCH_2:.2f}')
            if dist > OT_STATIC_SWITCH_2:
                self.switch_to_state(nac.LANE_FOLLOWING)
        else:
            self.error('ERROR: OVERTAKE: Wrong substate')

        self.curr_state.var1 = (sub_state, just_sub_switched)
        self.curr_state.var2 = dist_prev_manouver

    def overtaking_moving_car(self):
        self.activate_routines([])
        # states
        OT_SWITCHING_LANE = 1
        OT_LANE_FOLLOWING = 2
        OT_BESIDE_CAR = 3
        OT_WAITING_BEFORE_SWITCHING = 4
        OT_SWITCHING_BACK = 5

        if self.curr_state.just_switched:
            self.curr_state.var1 = (OT_SWITCHING_LANE, True)
            self.curr_state.var2 = self.car.encoder_distance             
            self.curr_state.just_switched = False
        sub_state, just_sub_switched = self.curr_state.var1
        dist_prev_manouver = self.curr_state.var2
        # 1st fase
        if sub_state == OT_SWITCHING_LANE:
            if just_sub_switched:
                self.car.drive_angle(-OVERTAKE_STEER_ANGLE)
                dist_prev_manouver = self.car.encoder_distance           
                self.car.drive_speed(OVERTAKE_MOVING_CAR_SPEED)
                just_sub_switched = False
            dist = self.car.encoder_distance - dist_prev_manouver            
            assert dist > -0.05
            print(f'Switching lane: {dist:.2f}/{OT_MOVING_SWITCH_1:.2f}')
            if dist > OT_MOVING_SWITCH_1:
                sub_state, just_sub_switched = OT_LANE_FOLLOWING, True
                dist_prev_manouver = self.car.encoder_distance               
        # 2nd fase             
        elif sub_state == OT_LANE_FOLLOWING:
            self.activate_routines([nac.FOLLOW_LANE])
            dist = self.car.encoder_distance - dist_prev_manouver            
            print(f'Following lane: {dist:.2f}/{OT_MOVING_LANE_FOLLOW:.2f}')
            if self.car.filtered_right_sonar_distance <= 0.5:  #0.5 is the distance from the lateral moving car 
                # Car is beside the other car
                sub_state, just_sub_switched = OT_BESIDE_CAR, True
                print ('We are beside the car')
            
        ###### New code to overtake moving car relying on sonar feedback 
        # it should check when the distance is low, means that the car is on the side
        # when the distance increase again, the car wait x second and then switch back lane   
        # 3rd fase
        elif sub_state == OT_BESIDE_CAR:
            self.activate_routines([nac.FOLLOW_LANE])
            dist = self.car.encoder_distance - dist_prev_manouver            
            print(f'Following lane besides car: {dist:.2f}/{OT_MOVING_LANE_FOLLOW:.2f}')
            if self.car.filtered_right_sonar_distance > 0.5: #0.3 distance from the lateral car
                # Car has been overtaken, switch back to the original lane after waiting some time
                sub_state, just_sub_switched = OT_WAITING_BEFORE_SWITCHING, True
                print ('We passed the car')
        # 4th fase
        elif sub_state == OT_WAITING_BEFORE_SWITCHING:
            print ('Waiting to go back to 2nd lane')
            if just_sub_switched == True:
                self.curr_state.var4 = time()
                just_sub_switched = False
            if self.car.filtered_right_sonar_distance <= 0.5:
                sub_state, just_sub_switched = OT_BESIDE_CAR, True
                print ('We are beside the car again')
            if ((time() - self.curr_state.var4) > 0.5): #waiting time to switch back to the 2nd lane just to be sure we are ahead enough
                sub_state, just_sub_switched = OT_SWITCHING_BACK, True                  
        ##### End of new code
        # 5th fase
        elif sub_state == OT_SWITCHING_BACK:
            if just_sub_switched:
                self.car.drive_angle(OVERTAKE_STEER_ANGLE)
                dist_prev_manouver = self.car.encoder_distance               
                just_sub_switched = False
            dist = self.car.encoder_distance - dist_prev_manouver            
            assert dist > -0.05
            print(f'Switching back: {dist:.2f}/{OT_MOVING_SWITCH_2:.2f}')
            if dist > OT_MOVING_SWITCH_2:
                self.switch_to_state(nac.LANE_FOLLOWING)
        else:
            self.error('ERROR: OVERTAKE: Wrong substate')
        self.curr_state.var1 = (sub_state, just_sub_switched)
        self.curr_state.var2 = dist_prev_manouver

    def tailing_car(self):

        dist = self.car.central_distance
        print(f"##################### DISTANCE {dist}")

        
        if dist > OBSTACLE_DISTANCE_THRESHOLD+0.05:  #TODO why is 0.05 here?!
            self.switch_to_state(nac.LANE_FOLLOWING)
            print('switching')
            if(nac.TESTING):
                print('entered')
                self.activate_routines([nac.FOLLOW_LANE, nac.DRIVE_DESIRED_SPEED, nac.DETECT_STOPLINE])
                self.run_routines()
        
        else:
            self.activate_routines([nac.FOLLOW_LANE, nac.DETECT_STOPLINE])
            if(nac.TESTING):
                print("!!!!!!!!!!!!!!!!!!!!!!!")
                self.run_routines()
            dist_to_drive = dist - TAILING_DISTANCE
            self.car.drive_distance(dist_to_drive)
            if self.conditions[nac.CAN_OVERTAKE] and not nac.TESTING:
                if self.conditions[nac.HIGHWAY]:
                    print("OVERTAKINGGGGGG")
                    self.switch_to_state(nac.OVERTAKING_MOVING_CAR)
                else:
                    if -0.05 < dist_to_drive < 0.05: # TODO: check the distance 
                        print("OVERTAKINGGGGGG STATICCCC")
                        self.switch_to_state(nac.OVERTAKING_STATIC_CAR)

            if self.detect.est_dist_to_stopline < STOPLINE_APPROACH_DISTANCE and self.routines[nac.DETECT_STOPLINE].active:
                print("SWITCHING TO APPROACHING STOPLINE")
                self.switch_to_state(nac.APPROACHING_STOPLINE)

    

    def parking(self):
        # Substates
        print(f'just switched: {self.curr_state.just_switched}')
        if self.curr_state.just_switched:
            # We just got in the parking state, we came from lane following,
            # we are reasonably close to the parking spot and we are not moving
            # self.curr_state.var1 will hold the parking substate,
            # the parking type, and if it has just changed state
            park_state = nac.LOCALIZING_PARKING_SPOT
            park_type = nac.S_PARK
            self.curr_state.var1 = (park_state, park_type, True)
            self.curr_state.var4 = self.car.encoder_distance
            self.curr_state.just_switched = False

        park_state, park_type, just_changed = self.curr_state.var1

        print(f'in the parking curr_state.var1: {self.curr_state.var1}')
        self.run_routines()
        print(f'ROUTINES:       {self.active_routines_names+ALWAYS_ON_ROUTINES}')

        #######################################################################
        # first state: localizing with precision the parking spot
        if park_state == nac.LOCALIZING_PARKING_SPOT:
            self.parking_localizing(just_changed, park_state, park_type)

        #######################################################################
        # second state: checking if there are parked cars in the parking spots
        elif park_state == nac.CHECKING_FOR_PARKED_CARS:
            self.parking_checking(just_changed, park_state, park_type)

        #######################################################################
        # STEP 0 -> ALIGN WITH THE PARKING SPOT
        elif park_state == nac.STEP0:
            self.parking_st0(just_changed, park_state, park_type)

        # S parking manouver
        elif park_state in [nac.S_STEP2, nac.S_STEP3, nac.S_STEP4, nac.S_STEP5, nac.S_STEP6, nac.S_STEP7]:
            self.car.drive(speed=0.0, angle=0)
            sleep(0.8)
            free_spot_R, free_spot_L, _ = self.curr_state.var2
            if free_spot_R:
                park.parallel_parking(self.car, nac.RIGHT_PARK)
            elif free_spot_L:
                park.parallel_parking(self.car, nac.LEFT_PARK)
            
            self.curr_state.var1 = (nac.PARK_END, park_type, True)
            # self.parking_s(just_changed, park_state, park_type)

        # end of manouver, go to next event
        elif park_state == nac.PARK_END:
            self.parking_end()
#######################################################################
    def parking_localizing(self, just_changed, park_state, park_type):
        print('LOCALIZING_PARKING_SPOT')
        self.activate_routines([nac.FOLLOW_LANE])

       # if(nac.TESTING):
         #   print('We arrived at the parking spot')
         #   self.car.drive_speed(0.0)
          #  self.curr_state.var1 = (nac.CHECKING_FOR_PARKED_CARS, park_type, True)
          #  print(f'curr_state.var1: {self.curr_state.var1}')
          #  return

        if just_changed:
            # We will use local positioning to localize the parking spot
            self.curr_state.var1 = (park_state, park_type, False)
            self.curr_state.var2  = True
            self.car.reset_rel_pose()
            
        car_est_pos = np.array([self.car.x_est, self.car.y_est])
        # one sample for every cm in the path
        park_index_on_path = int(self.next_event.dist*100)
        path_to_analyze = self.path_planner.path[max(0, park_index_on_path - SUBPATH_LENGTH_FOR_PARKING): min(park_index_on_path + SUBPATH_LENGTH_FOR_PARKING, len(self.path_planner.path))]
        car_idx_on_path = np.argmin(norm(path_to_analyze - car_est_pos, axis=1))
        park_index_on_path = SUBPATH_LENGTH_FOR_PARKING

        # print("path_to_analyze ", path_to_analyze)
        print("car_est_pos ", car_est_pos)
        print("car_idx_on_path ", car_idx_on_path)
        print("park_index_on_path ", park_index_on_path)
        print("self.car.dist_loc ", self.car.dist_loc)
        print("MAX_PARK_SEARCH_DIST ", MAX_PARK_SEARCH_DIST)

        if self.car.dist_loc < MAX_PARK_SEARCH_DIST:
            print('Behind parking spot')
            self.car.drive_speed(PARK_SEARCH_SPEED)
            if self.car.dist_loc > MIN_PARK_SEARCH_DIST:
                print('We arrived at the parking spot')
                self.car.drive_speed(0.0)
                self.curr_state.var1 = (nac.CHECKING_FOR_PARKED_CARS, park_type, True)
            else:
                print(f'getting closer...  dist: {self.car.dist_loc:.2f}/{MAX_PARK_SEARCH_DIST:.2f}')
        else:
            self.error('ERROR: PARKING: In front of parking spot, or maximum search distance reached')

    def parking_checking(self, just_changed, park_state, park_type):
        print('Checking for parked cars...')
        if just_changed:
            print('Activating routines')
            self.activate_routines([nac.FOLLOW_LANE])
            if not nac.TESTING:
                assert self.next_event.name == nac.PARKING_EVENT
            self.car.reset_rel_pose()
            # (free_spot_R, free_spot_L, checked_spots_counter (0 to 5))
            self.curr_state.var2 = (False, False, 0)
            self.car.drive_speed(PARK_MANOUVER_SPEED)
            self.curr_state.var1 = (park_state, park_type, False)

        if(nac.TESTING):
            self.activate_routines([nac.FOLLOW_LANE])
            #self.car.drive_speed(PARK_MANOUVER_SPEED)
            

        free_spot_R, free_spot_L, checked_spots_counter = self.curr_state.var2
        curr_dist = self.car.dist_loc

        print(f'current distance :{curr_dist}')
  
        if park_type == nac.S_PARK:
            dist_first_spot = DIST_SIGN_FIRST_S_SPOT
            dist_spots = DIST_S_SPOTS
            further_dist = FURTHER_DIST_S
        else:
            self.error('ERROR: PARKING: Unknown parking type!')
                
        if ((dist_first_spot + dist_spots*checked_spots_counter) <= curr_dist < (dist_first_spot + dist_spots*checked_spots_counter+0.1)):
            self.car.drive_speed(0.0)
            sleep(SLEEP_AFTER_STOPPING)
            # get right and left sonar distance
            right_sonar_dist = self.car.filtered_right_sonar_distance
            left_sonar_dist = self.car.filtered_left_sonar_distance
            print(f'Spot checked: {checked_spots_counter+1}')
            print(f'Right sonar {right_sonar_dist}')
            print(f'Left sonar {left_sonar_dist}')
            if right_sonar_dist < 0.5:
                print('Car in park_right')
                self.env.publish_obstacle(nac.STATIC_CAR_PARKING, self.car.x_est, self.car.y_est-0.4)
            else:
                free_spot_R = True
                print('Free parking right')
                self.curr_state.var1 = (nac.STEP0, park_type, True)
            if left_sonar_dist < 0.5:
                print('Car in park_left')
                self.env.publish_obstacle(nac.STATIC_CAR_PARKING, self.car.x_est, self.car.y_est+0.4)
            else:
                free_spot_L = True
                print('Free parking left')
                self.curr_state.var1 = (nac.STEP0, park_type, True)
            checked_spots_counter += 1
        
        if not (free_spot_L or free_spot_R):
            self.car.drive_speed(PARK_MANOUVER_SPEED)
            if checked_spots_counter >= 5:
                print('CHECKING_CARS: No free spot for parking')
                self.curr_state.var1 = (nac.PARK_END, park_type, True)  

        if dist_first_spot+(dist_spots*5)+further_dist + MAX_ERROR_ON_LOCAL_DIST < curr_dist:
            overshoot_distance = dist_first_spot + (dist_spots*5) + further_dist + MAX_ERROR_ON_LOCAL_DIST-curr_dist
            self.error(f'ERROR: PARKING: CHECKING_CARS: Overshoot distance, error: {overshoot_distance:.2f}')
         
        # update var2 at the end of every iteration
        self.curr_state.var2 = (free_spot_R, free_spot_L, checked_spots_counter)


    def parking_st0(self, just_changed, park_state, park_type):
        # we are standing still besides the free spot
        print('STEP0 -> Going forward...')

        free_spot_R, free_spot_L, _= self.curr_state.var2

        if just_changed:
            if not (free_spot_R or free_spot_L):
                self.switch_to_state(nac.LANE_FOLLOWING)
            else:
                self.activate_routines([nac.FOLLOW_LANE])
                self.car.reset_rel_pose()
                self.car.drive_speed(PARK_MANOUVER_SPEED)
                self.curr_state.var1 = (park_state, park_type, False)

        dist = self.car.dist_loc
        print(f'Distance: {dist}')
        dist_spots = DIST_FORWARD
        if dist > dist_spots:
            self.car.drive_speed(0.0)
            sleep(SLEEP_AFTER_STOPPING)
            print('Aligned for parking spot')
            park_state = nac.S_STEP2
            self.curr_state.var1 = (park_state, park_type, True)
        if dist > dist_spots + MAX_ERROR_ON_LOCAL_DIST:
            overshoot_distance = dist_spots + MAX_ERROR_ON_LOCAL_DIST - dist
            self.error(f'ERROR: PARKING: STEP0: Overshoot distance, error: {overshoot_distance:.2f}')


    def parking_end(self):
        self.switch_to_state(nac.LANE_FOLLOWING)
        self.go_to_next_event()

    def crosswalk_navigation(self):
        if nac.TESTING:
            self.activate_routines([nac.CONTROL_FOR_PEDESTRIAN])
            if not (self.flag_seen_pedestrian or self.car.central_distance < PEDESTRIAN_CONTROL_DISTANCE):
                self.activate_routines([nac.FOLLOW_LANE,
                                        nac.DRIVE_DESIRED_SPEED])
            self.run_routines()

        if self.curr_state.just_switched:
                self.curr_state.just_switched = False
                self.car.reset_rel_pose()

        print (f'flag {self.flag_seen_pedestrian}')
        print (f'in rect {self.flag_pedestrian_in_the_way}')
        print (f'lidar dist {self.car.central_distance}')
        print (f'on crosswalk {self.pedestrian_on_the_crosswalk}')


        if self.flag_seen_pedestrian or self.car.central_distance < PEDESTRIAN_CONTROL_DISTANCE or (time() - self.curr_state.start_time) < 10:
            self.car.drive_speed(0.0)
            self.activate_routines([nac.CONTROL_FOR_PEDESTRIAN])
            if not self.pedestrian_on_the_crosswalk:
                sleep(3)
            if self.flag_pedestrian_in_the_way or self.car.central_distance < PEDESTRIAN_CONTROL_DISTANCE:
                self.car.drive_speed(0.0)
                self.pedestrian_on_the_crosswalk = True
            else:
                self.car.drive_speed(self.desired_speed)
                self.switch_to_state(nac.LANE_FOLLOWING)
                self.go_to_next_event()


        else:
            self.car.drive_speed(self.desired_speed)

            if not nac.TESTING:
                print('in not testing')
                self.switch_to_state(nac.LANE_FOLLOWING)
                self.go_to_next_event()
   
        

    def tunnel_speed_curve(self):
        right_distance = hf.get_min_distance_in_range(self.car.lidar_angles,self.car.lidar_ranges, -95, -85)
        if right_distance < 0.7: 
            self.activate_routines([nac.DRIVE_DESIRED_SPEED, nac.CONTROL_FOR_CAR]) 
            self.run_routines()

            error = - 0.25 + right_distance # desired - actual distance [m]

            # Time delta for PI control
            current_time = time()
            if self.tunnel_last_time is None:
                delta_time = 0.0
                self.tunnel_last_time = current_time
            else:
                delta_time = current_time - self.tunnel_last_time
                self.tunnel_last_time = current_time

            # Update integral term with anti-windup clamping
            self.tunnel_integral_error += error * delta_time
            self.tunnel_integral_error = max(min(self.tunnel_integral_error, 0.8), -0.8) # Clamp integral

            # PI control
            steering_output = (150 * error) + (100 * self.tunnel_integral_error)
            steering_output = max(min(steering_output, 28), -28) # Clamp output
            self.car.drive_angle(steering_output)
            print(f"error: {error:.2f}, integral error: {self.tunnel_integral_error:.2f}, steering output: {steering_output:.2f}")
        else:
            # Reset the integral error when the car is not in the tunnel
            self.tunnel_integral_error = 0.0
            self.tunnel_last_time = None
            # Switch to lane following state
            self.switch_to_state(nac.LANE_FOLLOWING)
            self.go_to_next_event()


    def no_lane(self):
        travelled_distance = self.car.encoder_distance - self.curr_state.start_distance
        print(f'/---------------/ TRAVELLED DISTANCE = {travelled_distance:.2f}')
        if self.checkpoints[self.checkpoint_idx] in range(302, 331) or nac.TESTING:
            #print('in RIGHT NO LANE!!!!!!!!')

            if travelled_distance <= 4.0:
                self.activate_routines([nac.FOLLOW_LANE,
                                        nac.CONTROL_FOR_CAR,
                                        nac.DETECT_STOPLINE,
                                        nac.DRIVE_DESIRED_SPEED])
                self.run_routines()

            elif 4.0 < travelled_distance and travelled_distance < 4.5:
                self.activate_routines([nac.FOLLOW_LANE_LEFT,
                                nac.CONTROL_FOR_CAR,
                                nac.DETECT_STOPLINE,
                                nac.DRIVE_DESIRED_SPEED])
                self.run_routines()
            
            else:
                #self.NO_LANE_CAN_BE_ACTIVATED = False
                #self.conditions[nac.NO_LANE] = False
                self.switch_to_state(nac.LANE_FOLLOWING)
                nac.DONT_STOP_AT_NO_LANE_EVENT = True
                self.go_to_next_event()

        

        else:
            #print('in LEFT NO LANE!!!!!!!!')

            # KEEP LANE
            if travelled_distance <= 4.7:
                self.activate_routines([nac.FOLLOW_LANE,
                                        nac.CONTROL_FOR_CAR,
                                        nac.DETECT_STOPLINE,
                                        nac.DRIVE_DESIRED_SPEED])
                self.run_routines()
            
            # KEEP RIGHT WHEN YOU LOOSE LINE MARKER
            elif travelled_distance <= 5.2:
                self.activate_routines([nac.FOLLOW_LANE_RIGHT,
                                nac.CONTROL_FOR_CAR,
                                nac.DETECT_STOPLINE,
                                nac.DRIVE_DESIRED_SPEED])
                self.run_routines()

            # KEEP LANE IN THE DOTTED LINE SECTION
            elif travelled_distance <= 6.8:
                self.activate_routines([nac.FOLLOW_LANE,
                                nac.CONTROL_FOR_CAR,
                                nac.DETECT_STOPLINE,
                                nac.DRIVE_DESIRED_SPEED])
                self.run_routines()

            # KEEP RIGHT WHEN YOU LOOSE LINE MARKER
            elif travelled_distance <= 7.1:
                self.activate_routines([nac.FOLLOW_LANE_RIGHT,
                                nac.CONTROL_FOR_CAR,
                                nac.DETECT_STOPLINE,
                                nac.DRIVE_DESIRED_SPEED])
                self.run_routines()
            
            # SWITCH STATE TO LANE FOLLOWING & RESET THE FLAG NO_LANE_CAN_BE_ACTIVATED 
            else:
                #self.NO_LANE_CAN_BE_ACTIVATED = False
                #self.conditions[nac.NO_LANE] = False
                self.switch_to_state(nac.LANE_FOLLOWING)
                nac.DONT_STOP_AT_NO_LANE_EVENT = True
                self.go_to_next_event() 


    # =============== ROUTINES =============== #

    def follow_lane(self):
        e2, e3, point_ahead = self.detect.detect_lane(self.car.frame, SHOW_IMGS)
        # print("\nERROR e2 = ", e2)
        # print("ERROR e3 = ", e3)
        # print("ERROR point_ahead = ", point_ahead, "\n")
        print("In the lane follow!")
        hf.show_follow_lane(self, point_ahead, SHOW_IMGS)
        _, angle_ref = self.controller.get_control(e2, e3, 0, self.desired_speed, no_lane=False)
        angle_ref = np.rad2deg(angle_ref)
        self.car.drive_angle(angle_ref)
        

    def follow_roundabout(self):
        e3, point_ahead = self.detect.detect_roundabout_about(self.car.frame, SHOW_IMGS)
        hf.show_follow_lane(self, point_ahead, SHOW_IMGS)
        _, angle_ref = self.controller.get_control(0, e3, 0, self.desired_speed, no_lane=False)
        angle_ref = np.rad2deg(angle_ref)
        self.car.drive_angle(angle_ref)

    def follow_lane_right(self):
        e3, point_ahead = self.detect.detect_intersection_right(self.car.frame, SHOW_IMGS)
        hf.show_follow_lane(self, point_ahead, SHOW_IMGS)
        _, angle_ref = self.controller.get_control(0, e3, 0, self.desired_speed, no_lane=False)
        angle_ref = np.rad2deg(angle_ref)
        self.car.drive_angle(angle_ref)

    def follow_lane_left(self):
        e3, point_ahead = self.detect.detect_intersection_left(self.car.frame, SHOW_IMGS)
        hf.show_follow_lane(self, point_ahead, SHOW_IMGS)
        _, angle_ref = self.controller.get_control(0, e3, 0, self.desired_speed, no_lane=False)
        angle_ref = np.rad2deg(angle_ref)
        self.car.drive_angle(angle_ref)


    def detect_stopline(self):
        # update the variable self.detect.est_dist_to_stopline
        stopline_x, _, _ = self.detect.detect_stopline(self.car.frame, show_ROI=SHOW_IMGS)
        dist = stopline_x + 0.05

        past_detections = self.routines[nac.DETECT_STOPLINE].var2
        # -0.1 #network is more accurate in this range
        if dist < STOPLINE_APPROACH_DISTANCE-0.05 :  #0.05
            DETECTION_DEQUE_LENGTH = 50            # move to nac?
            SAMPLE_BEFORE_CONFIDENCE = 20          # move to nac?
            # var1 holds last detection time
            if self.routines[nac.DETECT_STOPLINE].var1 is not None:
                last_detection_time = self.routines[nac.DETECT_STOPLINE].var1
            else:
                last_detection_time = time() - 1.0
            curr_time = time()

            if curr_time - last_detection_time > 0.5:
                # var2 holds the list of past detections, reset
                self.routines[nac.DETECT_STOPLINE].var2 = past_detections = deque(maxlen=DETECTION_DEQUE_LENGTH)

            adapted_distance = dist + self.car.encoder_distance          

            past_detections.append(adapted_distance)
            if len(past_detections) > SAMPLE_BEFORE_CONFIDENCE:
                # NOTE consider also mean
                self.stopline_distance_median = np.median(past_detections)
            else:
                self.stopline_distance_median = None
            self.routines[nac.DETECT_STOPLINE].var1 = curr_time
        else:
            self.stopline_distance_median = None

    def slow_down(self):
        if np.abs(self.car.filtered_encoder_velocity - SLOW_DOWN_CONST*self.desired_speed) > 0.1:
            self.car.drive_speed(self.desired_speed*SLOW_DOWN_CONST)

    def accelerate(self):
        if np.abs(self.car.filtered_encoder_velocity < ACCELERATION_CONST*self.desired_speed):
            self.car.drive_speed(ACCELERATION_CONST*self.desired_speed)

    def control_for_signs(self): # we dont do, either do it or delete it 
        prev_sign = self.curr_sign
        if not self.conditions[nac.REROUTING]:
             # Use signs                # not implemented? <++>
            return  # TODO remove it and use better detection
            sign, _ = self.detect.detect_sign(self.car.frame,
                                              show_ROI=SHOW_IMGS,
                                              show_kp=SHOW_IMGS)
            if sign != nac.NO_SIGN and sign != self.curr_sign:
                self.curr_sign = sign

            # publish sign
            if self.curr_sign != prev_sign and self.curr_sign != nac.NO_SIGN:
                self.env.publish_obstacle(self.curr_sign, self.car.x_est, self.car.y_est)

    def control_for_car(self):
        # check for obstacles
        #print('[CONTROLING FOR CAR]')
        if self.routines[nac.CONTROL_FOR_CAR].var1 is not None:
            last_obstacle_dist = self.routines[nac.CONTROL_FOR_CAR].var1
        else:
            last_obstacle_dist = self.car.encoder_distance - 1.0                 
        curr_dist = self.car.encoder_distance                                    
        if curr_dist - last_obstacle_dist > MIN_DIST_BETWEEN_OBSTACLES:
            dist = self.car.central_distance
            print('DISTANCE:', dist)
            if dist < OBSTACLE_DISTANCE_THRESHOLD and not self.curr_state==nac.APPROACHING_STOPLINE and not self.curr_state==nac.WAITING_AT_STOPLINE:
                print('[### CAR DETECTED ###]')
                self.car.drive_speed(speed=self.desired_speed/10)
                obstacle = nac.CAR
                print(f'Obstacle: {obstacle}')
                self.routines[nac.CONTROL_FOR_CAR].var1 = curr_dist
                self.switch_to_state(nac.TAILING_CAR)
                

    def control_for_pedestrian(self):
        #print('[CONTROLING FOR PEDESTRIAN]')
        # check for pedestrian
        frame = self.car.frame

        # Resize the frame to match the preview resolution (faster display)
        frame_resized = self.car.frame.copy()

        # Convert the frame to HSV color space
        hsv = cv.cvtColor(frame_resized, cv.COLOR_RGB2HSV)

        # Define the range of pink in HSV (Hue range for pink: 140-170)
        lower_pink = np.array([120, 60, 60])  # Lower bound for pink
        upper_pink = np.array([170, 255, 255])  # Upper bound for pink

        # Create a mask for pink regions
        mask = cv.inRange(hsv, lower_pink, upper_pink)

        # Find contours in the mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour, assuming the pink object is the largest object
            largest_contour = max(contours, key=cv.contourArea)

            # Get the moments to find the centroid
            moments = cv.moments(largest_contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                # Draw a red dot at the centroid (Red is BGR: (0, 0, 255))
                cv.circle(frame_resized, (cx, cy), 10, (0, 0, 255), -1)  # Red in BGR
                
                # Define the rectangle centered in the image
                frame_height, frame_width = frame_resized.shape[:2]
                rect_width = 200  # Width of the rectangle
                rect_height = 200  # Height of the rectangle

                # Define the top-left and bottom-right corners of the rectangle
                rect_x1 = (frame_width - rect_width) // 2
                rect_y1 = (frame_height - rect_height) // 2
                rect_x2 = rect_x1 + rect_width
                rect_y2 = rect_y1 + rect_height

                # Draw the rectangle (Green in BGR: (0, 255, 0))
                cv.rectangle(frame_resized, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 2)

                # Check if the centroid is inside the rectangle
                if rect_x1 <= cx <= rect_x2 and rect_y1 <= cy <= rect_y2:
                    self.flag_pedestrian_in_the_way = True
                else:
                    self.flag_pedestrian_in_the_way =  False

                self.flag_seen_pedestrian = True
            else:
                self.flag_seen_pedestrian =  False

        print(f"Pedestrian :{self.flag_pedestrian_in_the_way}")

        #if flag_seen_pedestrian: 
            # start checking the lidar when we get close to the crosswalk

        # Display the resulting frame (now resized for faster performance)
        #cv.imshow("Camera", frame_resized)
        #cv.waitKey(1)
 

    def drive_desired_speed(self):
        if np.abs(self.car.filtered_encoder_velocity - self.desired_speed) > 0.1:
            self.car.drive_speed(self.desired_speed)

    # UPDATE CONDITIONS
    def update_state(self):
        """"
        This will update the conditions at every iteration, it is called at
        the end of a self.run
        """
        # deque of past imgs
        print('UPDATE STATE')
        if self.routines[nac.UPDATE_STATE].var1 is not None:
            prev_dist = self.routines[nac.UPDATE_STATE].var1
        else:
            prev_dist = self.car.encoder_distance                    
        curr_dist = self.car.encoder_distance                        
        print(f'VAR1: {self.routines[nac.UPDATE_STATE].var1}')
        print(f'DISTANCE CHANGE: {prev_dist-curr_dist}')
        if np.abs(prev_dist-curr_dist) > DISTANCES_BETWEEN_FRAMES:
            self.past_frames.append(self.car.frame)
            prev_dist = curr_dist
        self.routines[nac.UPDATE_STATE].var1 = prev_dist

        self.car_dist_on_path += curr_dist - prev_dist
        # print('PATH: ', self.path_planner.path)
        print('DIST ON PATH: ', self.car_dist_on_path)

        # HIGHWAY
        '''
            This condition is turned False every time we hit a stopline (inside approaching_stopline function)
            So this only works in one way of the highway
        '''
        #TODO: implement that it can work on both  ways!
        if (self.next_event.name == nac.HIGHWAY_ENTRANCE_EVENT) and (self.car.filtered_left_sonar_distance <= 0.5):
            # self.conditions[nac.HIGHWAY] = str(self.checkpoints[self.checkpoint_idx]) in self.path_planner.highway_nodes and self.car_dist_on_path < 9.5
            self.conditions[nac.HIGHWAY] = True  
        
        # NO_LANE
        '''
            This condition is turned False every time we hit a stopline
        '''
        #if (self.next_event.name == nac.NO_LANE_EVENT):
        #    self.conditions[nac.NO_LANE] = True

        # CAN_OVERTAKE
        '''
            Very ugly solution used in 2024, please do better in terms of when to overtake
        '''
        print(self.stopline_counter)
        self.conditions[nac.CAN_OVERTAKE] = (self.conditions[nac.HIGHWAY] or self.stopline_counter in OVERTAKE_COUNTER)

    # ===================== STATE MACHINE MANAGEMENT ===================== #
    def run(self):
        print('==========================================================================')
        print(f'CHECKPOINT:     {self.checkpoints[self.checkpoint_idx]}')
        print(f'STATE:          {self.curr_state}')
        print(f'PREV_EVENT:     {self.prev_event}')
        print(f'UPCOMING_EVENT: {self.next_event}')
        print(f'ROUTINES:       {self.active_routines_names+ALWAYS_ON_ROUTINES}')
        print(f'CONDITIONS:     {self.conditions}')
        print(f'next next event: {self.second_next_event}')
        print('==========================================================================')
        self.run_current_state()
        print(f'car.yaw: {self.car.yaw}')
        print(f'car.yaw_loc: {self.car.yaw_loc}')
        print('==========================================================================')
        print(f'car x est: {self.car.x_est}')
        print(f'car y est: {self.car.y_est}')
        print(f'CHECKPOINT IDX: {self.checkpoint_idx}')
        print(f'CHECKPOINT lenght: {len(self.checkpoints)}')
        print('==========================================================================')
        print(f'stopline counter: {self.stopline_counter}')
        
        self.run_routines()

        # =============== Publish to dashboard ==================== #
        if True:
            self.car.publish_next_event(str(self.next_event))
            self.car.publish_prev_event(str(self.prev_event))
            self.car.publish_current_state(str(self.curr_state))
            routines_str = ";".join(self.active_routines_names+ALWAYS_ON_ROUTINES)
            self.car.publish_routines(str(routines_str))
            self.car.publish_conditions(self.conditions)


        
        # print(f'CURR_SIGN: {self.curr_sign}')  



    def run_current_state(self):
        self.curr_state.run()

    def run_routines(self):
        for k, r in self.routines.items():
            if r.active:
                r.run()
        for k in ALWAYS_ON_ROUTINES:
            self.routines[k].run()

    def activate_routines(self, routines_to_activate):
        """
        routines_to_activate are a list of strings (routines)
        ex: ['follow_lane', 'control_for_signs']
        """
        assert all([r in self.routines.keys() for r in routines_to_activate]), 'ERROR: activate_routines: routines_to_activate contains invalid routine'
        self.active_routines_names = []
        for k, r in self.routines.items():
            r.active = k in routines_to_activate
            if r.active:
                self.active_routines_names.append(k)

    def add_routines(self, routines):
        """
        add routines to the active routines without overwriting the other
        ex: ['follow_lane', 'control_for_signs']
        """
        assert all([r in self.routines.keys() for r in routines]), 'ERROR: add_routines: routines_to_activate contains invalid routine'
        for k in routines:
            self.routines[k].active = True

    def switch_to_state(self, to_state, interrupt=False):
        """
        to_state is the string of the desired state to switch to
        ex: 'lane_following'
        """
        assert to_state in self.states, f'{to_state} is not a valid state'
        self.prev_state = self.curr_state
        self.curr_state = self.states[to_state]
        for k, s in self.states.items():
            s.active = k == to_state
        if not interrupt:
            self.curr_state.start_time = time()
            self.curr_state.start_position = np.array([self.car.x_est, self.car.y_est])  # maybe another position
            self.curr_state.start_distance = self.car.encoder_distance           
            self.curr_state.interrupted = False
        else:
            self.curr_state.interrupted = True
        self.curr_state.just_switched = True

    # used only once, could be handeled differently <++>
    def switch_to_prev_state(self):
        self.switch_to_state(self.prev_state.name)

    def go_to_next_event(self):
        """
        Switches to the next event on the path
        """
        self.prev_event = self.next_event
        if self.event_idx == len(self.events):
            # no more events, for now
            pass
        else:
            self.next_event = self.events[self.event_idx]
            self.event_idx += 1
            ## Not sure of its usefulness, but we add it for now
            if self.event_idx == len(self.events):
                pass
            else:
                self.second_next_event = self.events[self.event_idx]

    def next_checkpoint(self):
        self.checkpoint_idx += 1
        # check if it's last
        if self.checkpoint_idx < (len(self.checkpoints)-1):
            # update events
            self.prev_event = self.next_event  # deepcopy(self.next_event)
            pass
        else:
            # it was the last checkpoint
            # TODO : test this implemetation of PUTA, is should make the car stop only at stoplines, not in the middle of the road 
            #if nac.PUTA:
            #    nac.PUTA = False
            #    self.switch_to_state(nac.LANE_FOLLOWING)
            #    pass

            print('Reached last checkpoint...\nExiting...')      
            self.car.drive(speed=0.0, angle=0.0)
            ###
            enc_start = self.car.encoder_distance              #BFMC_2024 <++> Commented 2025
            if self.checkpoints[-1] == END_NODE:                   
                extra_dist = 0.00                                 
                self.car.drive(speed=0.1, angle=0.0)
                while self.car.encoder_distance - enc_start < extra_dist:
                    sleep(0.1)
            else:
                self.car.drive(speed=0.0, angle=0.0)
            ###
            self.car.stop()
            sleep(3)
            cv.destroyAllWindows() if SHOW_IMGS else None
            exit()

    def create_sequence_of_events(self, events):
        """
        events is a list of strings (events)
        ex: ['lane_following', 'control_for_signs']
        """
        to_ret = []
        for e in events:
            name = e[0]
            dist = e[1]
            point = e[2]
            path_ahead = e[3]  # path in global coordinates
            #print(f"Name:           {name}")
            #print(f"Distance:       {dist:.2f}")
            #print(f"Point:          [{point[0]:.2f}, {point[1]:.2f}]")
            #print(f'Path ahead:     {path_ahead}')

            if path_ahead is not None:
                loc_path = path_ahead - point
                # get yaw of the stopline
                assert path_ahead.shape[0] > 10, f'path_ahead is too short: {path_ahead.shape[0]}, {events}'     
                path_first_10 = path_ahead[:10]
                diff10 = path_first_10[1:] - path_first_10[:-1]
                yaw_raw = np.median(np.arctan2(diff10[:, 1], diff10[:, 0]))
                yaw_stopline = hf.get_yaw_closest_axis(yaw_raw)
                loc_path = loc_path @ hf.rot_matrix(yaw_stopline)
                path_to_ret = loc_path
                curv = hf.get_curvature(path_ahead)
                #Debugging
                #print(f'yaw_stopline: {yaw_stopline}, name: {name}, curv: {curv}')
                len_path_ahead = 0.01*len(path_ahead)
            else:
                path_to_ret = None
                curv = None
                yaw_stopline = None
                len_path_ahead = None

            # going straight is ~0.15, right is ~ -0.15
            if name == nac.HIGHWAY_EXIT_EVENT and curv > 0.05:
                pass  # skip this case
            else:
                event = Event(name, dist, point, yaw_stopline, path_to_ret,
                              len_path_ahead, curv)
                to_ret.append(event)
        # add end of path event
        ee_point = self.path_planner.path[-1]
        end_event = Event(nac.END_EVENT, dist=0.0, point=ee_point)
        to_ret.append(end_event)
        return to_ret

    # Utility functions
    # should be moved to help functions? <++>
    def get_frames_in_range(self, start_dist, end_dist=0.0):
        len_past_frames = len(self.past_frames)
        print(f'len_past_frames: {len_past_frames}')
        idx_start = int(round(start_dist/DISTANCES_BETWEEN_FRAMES))
        idx_end = int(round(end_dist/DISTANCES_BETWEEN_FRAMES))
        print(f'idx_start: {idx_start}, idx_end: {idx_end}')
        assert idx_start < len_past_frames and idx_end >= 0
        # reverse the idxs
        idx_start = len_past_frames-1 - idx_start
        idx_end = len_past_frames-1 - idx_end
        print(f'idx_start: {idx_start}, idx_end: {idx_end}')
        # convert self.past_frames to a list
        past_frames = list(self.past_frames)
        return past_frames[idx_start:idx_end]

    # DEBUG
    def error(self, error_msg):
        print(error_msg)
        self.car.stop()
        sleep(3)
        exit()

