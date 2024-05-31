
"""
A simple example of an animated plot... In 3D!
"""
import numpy as np
import matplotlib.pyplot as plt
from vehicle_simulator.vehicle_models.fixed_wing_model import FixedWingModel
from vehicle_simulator.vehicle_models.fixed_wing_parameters import FixedWingParameters
from vehicle_simulator.vehicle_controllers.fixed_wing_autopilot import FixedWingControlParameters, FixedWingAutopilot
from vehicle_simulator.vehicle_controllers.fixed_wing_trajectory_tracker import FixedWingTrajectoryTracker
from vehicle_simulator.vehicle_controllers.bspline_trajectory_manager import SplineTrajectoryManager
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator
from vehicle_simulator.vehicle_simulators.fixed_wing_trajectory_tracking_simulator import FixedWingTrajectoryTrackingSimulator
from vehicle_simulator.vehicle_models.helper_functions import euler_to_quaternion



order = 3
gravity = 9.8
# max_roll = np.radians(25)
# desired_airspeed = 20
# max_pitch = np.radians(15)
# max_curvature = gravity*np.tan(max_roll)/(desired_airspeed**2)
# max_incline_angle = max_pitch
# max_incline = np.tan(max_incline_angle)
# max_incline = 0.2


control_points = np.array([[ 686.57019026,  593.2731686,   540.33713535,  525.53051979,  548.61493526,
   593.65942791,  639.61999546,  666.47527422,  656.61705212,  608.24973767,  510.38399719],
 [ -90.33885816, -104.83057092,  -90.33885816 , -47.2410361 ,   24.89631382,
   104.41551941,  183.4142707,   258.17307369,  299.33762797,  312.29095819,  299.33762797],
 [ 299.96306499,  300.01846751,  299.96306499,  300.2042128,   299.9058107,
   300.65733106,  300.60391714,  300.06318591,  300.1628035,   299.91859825,  300.1628035 ]])
scale_factor = 3.6558263719174797

bspline_eval = BsplineEvaluator(order)
start_velocity = bspline_eval.get_velocity_vector(0, control_points[:,0:4], scale_factor)
start_position = bspline_eval.get_position_vector(0, control_points[:,0:4], scale_factor)
control_point_list = [control_points]
scale_factor_list = [scale_factor]

fixed_wing_parameters = FixedWingParameters()
control_parameters = FixedWingControlParameters()
# Attaching 3D axis to the figure
fig = plt.figure()
ax = plt.axes(projection='3d')
fig.add_axes(ax)
north = start_position.item(0)
east = start_position.item(1)
down = start_position.item(2)
quat = euler_to_quaternion(0,0.05,np.pi)
u = 20
v = 0
w = 0
e0 = quat.item(0)
e1 = quat.item(1)
e2 = quat.item(2)
e3 = quat.item(3)
p = 0
q = 0
r = 0
wingspan = 2
fuselage_length = 2
state0 = np.array([north, east, down,  u, v, w,
                      e0,   e1,   e2, e3, p, q, r])

plane_model = FixedWingModel(ax, fixed_wing_parameters,
                  wingspan = wingspan, fuselage_length = fuselage_length,
                    state = state0)
autopilot = FixedWingAutopilot(control_parameters)
trajectory_tracker = FixedWingTrajectoryTracker(order, p_gain = 2, i_gain = 0.1, d_gain = 2, \
                                                feedforward_tolerance = 5, integrator_tolerance= 5, 
                                                start_position = start_position,
                                                fixed_wing_parameters = fixed_wing_parameters)
trajectory_manager = SplineTrajectoryManager(control_point_list, scale_factor_list, start_time=0,order=order)

wing_sim = FixedWingTrajectoryTrackingSimulator(plane_model=plane_model, plane_autopilot= autopilot,
                                                trajectory_tracker=trajectory_tracker, trajectory_manager=trajectory_manager)


vehicle_trajectory_data, trajectory_data = wing_sim.run_simulation()

