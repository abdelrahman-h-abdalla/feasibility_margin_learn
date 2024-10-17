"""
Created on Tue Jun 12 10:54:31 2018

@author: Romeo Orsolino
"""

import numpy as np

from numpy import array
from copy import deepcopy
import random
from jet_leg_common.jet_leg.computational_geometry.math_tools import Math
from jet_leg_common.jet_leg.optimization import nonlinear_projection
from jet_leg_common.jet_leg.computational_geometry.computational_geometry import ComputationalGeometry
from jet_leg_common.jet_leg.computational_geometry.iterative_projection_parameters import IterativeProjectionParameters
from jet_leg_common.jet_leg.optimization.jacobians_kinematics import KinematicJacobians
import time

import matplotlib.pyplot as plt
from jet_leg_common.jet_leg.plotting.plotting_tools import Plotter

plt.close('all')
plotter = Plotter()
math = Math()

''' Set the robot's name (either 'hyq', 'hyqreal', 'anymal_boxy' or 'anymal_coyote')'''
robot_name = 'hyqreal'

'''
possible constraints for each foot:
 ONLY_ACTUATION = only joint-torque limits are enforces
 ONLY_FRICTION = only friction cone constraints are enforced
 FRICTION_AND_ACTUATION = both friction cone constraints and joint-torque limits
'''
constraint_mode_IP = ['FRICTION_AND_ACTUATION',
                      'FRICTION_AND_ACTUATION',
                      'FRICTION_AND_ACTUATION',
                      'FRICTION_AND_ACTUATION']

comWF = np.array([.0, 0.0, 0.0])

''' stanceFeet vector contains 1 is the foot is on the ground and 0 if it is in the air'''
stanceFeet = [1, 1, 1, 1]

randomSwingLeg = random.randint(0, 3)
tripleStance = False  # if you want you can define a swing leg using this variable
if tripleStance:
    print('Swing leg', randomSwingLeg)
    stanceFeet[randomSwingLeg] = 0
print('stanceLegs ', stanceFeet)

projection = nonlinear_projection.NonlinearProjectionBretl(robot_name)

'''You now need to fill the 'params' object with all the relevant 
    informations needed for the computation of the IP'''
params = IterativeProjectionParameters(robot_name)
""" contact points in the World Frame"""
LF_foot = np.array([0.44, 0.34, -0.55])  # Starting configuration w.o. trunk controller
RF_foot = np.array([0.44, -0.34, -0.55])
LH_foot = np.array([-0.44, 0.34, -0.55])
RH_foot = np.array([-0.44, -0.34, -0.55])

contactsWF = np.vstack((LF_foot, RF_foot, LH_foot, RH_foot))

''' joint position limits for each leg (this code assumes a hyq-like design, i.e. three joints per leg)
HAA = Hip Abduction Adduction
HFE = Hip Flextion Extension
KFE = Knee Flextion Extension
'''
# LF_q_lim_max = [0.44, 1.2217, -0.3491]  # HAA, HFE, KFE
# LF_q_lim_min = [-1.22, -0.8727, -2.4435]  # HAA, HFE, KFE
# RF_q_lim_max = [0.44, 1.2217, -0.3491]  # HAA, HFE, KFE
# RF_q_lim_min = [-1.22, -0.8727, -2.4435]  # HAA, HFE, KFE
# LH_q_lim_max = [0.44, 0.8727, 2.4435]  # HAA, HFE, KFE
# LH_q_lim_min = [-1.22, -1.2217, 0.3491]  # HAA, HFE, KFE
# RH_q_lim_max = [0.44, 0.8727, 2.4435]  # HAA, HFE, KFE
# RH_q_lim_min = [-1.22, -1.2217, 0.3491]  # HAA, HFE, KFE
LF_q_lim_max = [0.401, 2.181, -0.770]  # HAA, HFE, KFE
LF_q_lim_min = [-0.733, 0.262, -2.770]  # HAA, HFE, KFE
RF_q_lim_max = [0.401, 2.181, -0.770]  # HAA, HFE, KFE
RF_q_lim_min = [-0.733, 0.262, -2.770]  # HAA, HFE, KFE
LH_q_lim_max = [0.401, 2.181, -0.770]  # HAA, HFE, KFE
LH_q_lim_min = [-0.733, 0.262, -2.770]  # HAA, HFE, KFE
RH_q_lim_max = [0.401, 2.181, -0.770]  # HAA, HFE, KFE
RH_q_lim_min = [-0.733, 0.262, -2.770]  # HAA, HFE, KFE
joint_limits_max = np.array([LF_q_lim_max, RF_q_lim_max, LH_q_lim_max, RH_q_lim_max])
joint_limits_min = np.array([LF_q_lim_min, RF_q_lim_min, LH_q_lim_min, RH_q_lim_min])

params.setContactsPosWF(contactsWF)


start = time.time()

# params.useContactTorque = True
params.useInstantaneousCapturePoint = False
params.setCoMPosWF(comWF)
params.setActiveContacts(stanceFeet)
params.setJointLimsMax(joint_limits_max)
params.setJointLimsMin(joint_limits_min)
com_check = None

params_com_x = deepcopy(params)
params_com_y = deepcopy(params)
params_com_z = deepcopy(params)

polygon, computation_time = projection.project_polytope(params, com_check, 20. * np.pi / 180, 0.03)

jac = KinematicJacobians(robot_name)


'''Margin'''
delta_pos_range = 0.3
num_of_tests = 50
delta_pos_range_vec = np.linspace(-delta_pos_range/2.0, delta_pos_range/2.0, num_of_tests)

# jac_com_pos_x_learnt = np.array([0.1318714921362698, 0.13494136813096702, 0.12848171521909535, 0.11865288577973843, 0.11080556083470583, 0.08709925645962358, 0.0803604807733791, 0.06896544806659222, 0.0429702396504581, 0.03134992532432079, 0.03038486884906888, 0.03533362550660968, 0.008302588044898584, -0.026188875315710902, -0.07204141980037093, -0.05818244256079197, -0.04362405283609405, -0.07244523259578273, -0.0808367938734591, -0.09529191441833973, -0.11372006189776585, -0.1328001539222896, -0.09160945890471339, -0.05243261705618352, -0.03810568095650524])
# jac_com_pos_y_learnt = np.array([0.04935239255428314, 0.06873649288900197, 0.0826681056059897, 0.07997685950249434, 0.07442796751274727, 0.07327371020801365, 0.06951900944113731, 0.06693135830573738, 0.06828382331877947, 0.0752132092602551, 0.09530838648788631, 0.07003822131082416, 0.04577533807605505, -0.04751181602478027, -0.07932630335562862, -0.10031966646783985, -0.1113769430667162, -0.10650849109515548, -0.11407906468957663, -0.09290009504184127, -0.06700065149925649, -0.052620375994592905, -0.045468409545719624, -0.046464577317237854, -0.04885811358690262])
# pos_margin_x_learnt_cpp = np.array([0.0632483, 0.0819287, 0.100212, 0.115994, 0.129758, 0.142651, 0.156272, 0.172334, 0.188274, 0.197773, 0.203808, 0.207759, 0.207001, 0.200641, 0.189896, 0.181941, 0.173401, 0.160104, 0.14332, 0.123397, 0.105915, 0.0933502, 0.0846915, 0.0798133, 0.0794336])
# pos_margin_y_learnt_cpp = np.array([-0.0345978, -0.0186792, 0.00161308, 0.0228501, 0.0453764, 0.0684856, 0.0919289, 0.113886, 0.134276, 0.154736, 0.175587, 0.195035, 0.207001, 0.198857, 0.185728, 0.164158, 0.139759, 0.113181, 0.0898839, 0.0696588, 0.0515738, 0.0362046, 0.0208963, 0.00610294, -0.00788821])

print("COM")
print("DELTA X")
pos_margin_x, jac_com_pos_x = jac.plotMarginAndJacobianWrtComPosition(params_com_x, delta_pos_range_vec, 0) # dm / dx
print("Jac_x:", jac_com_pos_x)
print("DELTA Y")
pos_margin_y, jac_com_pos_y = jac.plotMarginAndJacobianWrtComPosition(params_com_y, delta_pos_range_vec, 1) # dm / dy
print("Jac_y:", jac_com_pos_y)
print("DELTA Z")
# pos_margin_z, jac_com_pos_z = jac.plotMarginAndJacobianWrtComPosition(params_com_z, delta_pos_range_vec, 2) # dm / dz

# print("pos_margin_x:", pos_margin_x)
# print("pos_margin_y:", pos_margin_y)


'''Margin LH Swing'''
# 50 files, 512 epoch
# pos_margin_x_LH_learnt = np.array([-0.25780676150321963, -0.24563453793525697, -0.22590853643417358, -0.20267337369918823, -0.17867887938022614, -0.156468896985054, -0.1371808955669403, -0.12148259997367859, -0.10309817469120026, -0.08322921216487884, -0.06287937051057815, -0.04318124932050705, -0.020914322018623352, 0.0027260720059275627, 0.023533410608768464, 0.04136568135023117, 0.058880140364170074, 0.08065811026096344, 0.09480441731214523, 0.09400533610582351, 0.08878470486402512, 0.07491239100694656, 0.05714618879556656, 0.042898496389389036, 0.03487420678138733])
# pos_margin_y_LH_learnt = np.array([-0.03216588664054871, -0.01746982669830322, 0.0006292000338435173, 0.02011477106809616, 0.035570698499679566, 0.04241160982847214, 0.047300641357898715, 0.055399547517299655, 0.059412420094013214, 0.04836428117752075, 0.021434447944164277, -0.0022347179129719734, -0.020914322018623352, -0.04213895970582962, -0.06622177916765214, -0.08609677559137345, -0.10546432918310165, -0.12524207532405854, -0.1409557864665985, -0.14735884988307954, -0.14698892962932586, -0.14462185549736023, -0.14051389610767365, -0.13610305094718933, -0.1335275824069977])

stanceFeet = [1, 0, 1, 1]
params.setActiveContacts(stanceFeet)
params_com_x_LH = deepcopy(params)
params_com_y_LH = deepcopy(params)
params_com_z_LH = deepcopy(params)
pos_margin_x_LH, jac_com_pos_x_LH = jac.plotMarginAndJacobianWrtComPosition(params_com_x_LH, delta_pos_range_vec, 0) # dm / dx
pos_margin_y_LH, jac_com_pos_y_LH = jac.plotMarginAndJacobianWrtComPosition(params_com_y_LH, delta_pos_range_vec, 1) # dm / dy
# pos_margin_z_LH, jac_com_pos_z_LH = jac.plotMarginAndJacobianWrtComPosition(params_com_z_LH, delta_pos_range_vec, 2) # dm / dz

### Plotting

### X axis
fig1 = plt.figure(1)
fig1.suptitle("Stability margin")
plt.subplot(221)
plt.plot(delta_pos_range_vec, pos_margin_x, 'g', markersize=15, label='CoM')
# plt.plot(delta_pos_range_vec, pos_margin_x_learnt, 'b', markersize=15, label='CoM')
plt.grid()
plt.xlabel("$c_x$ [m]")
plt.ylabel("m [m]")
plt.title("CoM X pos margin")

plt.subplot(222)
plt.plot(delta_pos_range_vec, pos_margin_y, 'g', markersize=15, label='CoM')
# plt.plot(delta_pos_range_vec, pos_margin_y_learnt, 'b', markersize=15, label='CoM')
plt.grid()
plt.xlabel("$c_y$ [m]")
plt.ylabel("m [m]")
plt.title("CoM Y pos margin")

plt.subplot(223)
plt.plot(delta_pos_range_vec, pos_margin_x_LH, 'g', markersize=15, label='CoM')
# plt.plot(delta_pos_range_vec, pos_margin_x_LH_learnt, 'b', markersize=15, label='CoM')
plt.grid()
plt.xlabel("$c_x$ [m]")
plt.ylabel("m [m]")
plt.title("CoM X pos margin LH swing")

plt.subplot(224)
plt.plot(delta_pos_range_vec, pos_margin_y_LH, 'g', markersize=15, label='CoM')
# plt.plot(delta_pos_range_vec, pos_margin_y_LH_learnt, 'b', markersize=15, label='CoM')
plt.grid()
plt.xlabel("$c_y$ [m]")
plt.ylabel("m [m]")
plt.title("CoM Y pos margin LH swing")




# Jacobians
fig2 = plt.figure(2)
fig2.suptitle("Gradients")

plt.subplot(121)
plt.plot(delta_pos_range_vec, jac_com_pos_x[0], 'g', markersize=15, label='learned (backprop)')
# plt.plot(delta_pos_range_vec, jac_com_pos_x_learnt, 'b', markersize=15, label='learned (backprop)')
plt.grid()
plt.xlabel("$c_x$ [m]")
plt.ylabel("$\delta m/  \delta c_{x}$")
# plt.title("CoM X pos margin")
# ### Y axis
# fig2 = plt.figure(2)
# fig2.suptitle("Analytic stability margin")
# plt.subplot(231)
# plt.plot(delta_pos_range_vec, pos_margin_y, 'g', markersize=15, label='CoM')
# plt.grid()
# plt.xlabel("$c_y$ [m]")
# plt.ylabel("m [m]")
# plt.title("CoM Y pos margin")

plt.subplot(122)
plt.plot(delta_pos_range_vec, jac_com_pos_y[1], 'g', markersize=15, label='learned (backprop)')
# plt.plot(delta_pos_range_vec, jac_com_pos_y_learnt, 'b', markersize=15, label='learned (backprop)')
plt.grid()
plt.xlabel("$c_y$ [m]")
plt.ylabel("$\delta m/  \delta c_{y}$")
# plt.subplot(234)
# plt.plot(delta_pos_range_vec, jac_com_pos_y[1,:], 'g', markersize=15, label='CoM')
# plt.grid()
# plt.xlabel("$c_y$ [m]")
# plt.ylabel(" $ \delta m/  \delta c_y$")
# plt.title("CoM Y pos jacobian")

fig3 = plt.figure(3)
fig3.suptitle("Nominal Reachable Region")

nc = np.sum(stanceFeet)
stanceID = params.getStanceIndex(stanceFeet)

print("Projection:", polygon)
for j in range(0, nc):  # this will only show the contact positions and normals of the feet that are defined to be in stance
    idx = int(stanceID[j])
    ''' The black spheres represent the projection of the contact points on the same plane of the feasible region'''
    h5 = plt.plot(contactsWF[idx, 0], contactsWF[idx, 1], 'ko', markersize=15, label='stance feet')
h6 = plotter.plot_polygon(np.transpose(polygon), '--b', 5, 'Iterative Projection')

plt.show()