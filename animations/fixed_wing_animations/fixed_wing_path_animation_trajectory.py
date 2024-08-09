
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
gravity = 9.8
max_velocity = 28 #m/s
min_velocity = 16 #m/s
max_roll = np.radians(25)
max_centr_accel = gravity*np.tan(max_roll)
max_tang_accel = 5
min_tang_accel = -1.5

# constrained curvature and tange accel 28 m/s
# control_points = np.array([[ 770.97554794,  600.15676313,  428.39739953,  383.62744434,  501.58016088,
#    693.56792611,  839.91077866,  829.0830777,   813.79780047 , 770.75819338,
#    600.26544037,  428.18004515],
#  [-154.78908565 , -72.60545718, -154.78908565 ,-341.38959488, -491.90995326,
#   -493.11767462 ,-349.12348753, -100.32523817,  159.4482591,   345.26423442,
#    427.36788279 , 345.26423442],
#                  [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]])
# scale_factor =  6.2

# # # constrained by centripetal acceleration 28 m/s
# control_points = np.array([[ 715.98351396,  589.71080246 , 525.17327619 , 505.85159262 , 534.68784476,
#    587.10141494,  637.12343119 , 679.54258285 , 665.3699644,   615.01757738,
#    474.55972608],
#  [ -87.80931312, -106.09534344 , -87.80931312,  -27.84228043,   71.07603821,
#    159.83634307 , 249.96576717 , 334.3049774 ,  393.55128752,  403.22435624,
#    393.55128752],
#    [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]])
# scale_factor =  3.66942764608861

# # constrained by tangential acceleration 28 m/s
control_points = np.array([[ 6.74180876e+02 , 5.98367396e+02 , 5.32349539e+02,  5.46900072e+02,
   5.70990558e+02 , 5.95275311e+02 , 6.20379897e+02 , 6.49796070e+02,
   6.59421435e+02,  6.05747117e+02 , 5.17590097e+02],
 [-7.44038079e+01, -1.12798096e+02, -7.44038079e+01,  5.67602370e-01,
   7.30381386e+01,  1.45444647e+02 , 2.17570932e+02,  2.88019527e+02,
   3.63781253e+02 , 4.18109373e+02 , 3.63781253e+02],
   [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]])
scale_factor =  2.727525723915155

# constrained by centripetal acceleration and tangential acceleration 28 m/s
# control_points = np.array([[ 707.21550357,  596.95093223 , 504.98076752,  471.23380617  ,503.09400229,
#    580.1442987,   669.57950578,  710.43559646,  683.61557934,  608.75089435,
#    481.38084327],
#  [ -77.79432819, -111.10283591,  -77.79432819,   -7.2358383 ,   84.87869917,
#    161.84732501,  223.91555903 , 311.05299024,  384.2475073,   407.87624635, 384.2475073 ],
#    [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]])
# scale_factor =  3.889129538873446

print(np.shape(control_points))

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
quat = euler_to_quaternion(0,0,np.pi)
u = np.linalg.norm(start_velocity)
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
trajectory_tracker = FixedWingTrajectoryTracker(order, p_gain = 6, i_gain = 0.1, d_gain = 4, \
                                                feedforward_tolerance = 5, integrator_tolerance= 5, 
                                                start_position = start_position,
                                                fixed_wing_parameters = fixed_wing_parameters,
                                                max_velocity=max_velocity,
                                                min_velocity=min_velocity)
trajectory_manager = SplineTrajectoryManager(control_point_list, scale_factor_list, start_time=0,order=order)

wing_sim = FixedWingTrajectoryTrackingSimulator(plane_model=plane_model, plane_autopilot= autopilot,
                                                trajectory_tracker=trajectory_tracker, trajectory_manager=trajectory_manager,
                                                )

vehicle_trajectory_data, trajectory_data = wing_sim.run_simulation(graphic_scale=15, instances_per_plot=11)

wing_sim.plot_simulation_analytics(vehicle_trajectory_data, trajectory_data, max_velocity, max_centr_accel,
                                   max_tang_accel, min_tang_accel)

