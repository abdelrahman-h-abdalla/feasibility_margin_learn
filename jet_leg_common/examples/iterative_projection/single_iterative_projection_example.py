# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:54:31 2018

@author: Romeo Orsolino
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
from numpy import array
from scipy.spatial.transform import Rotation as Rot
from jet_leg_common.jet_leg.plotting.plotting_tools import Plotter
import random
from jet_leg_common.jet_leg.computational_geometry.math_tools import Math
from jet_leg_common.jet_leg.dynamics.computational_dynamics import ComputationalDynamics
from jet_leg_common.jet_leg.dynamics.instantaneous_capture_point import InstantaneousCapturePoint
from jet_leg_common.jet_leg.computational_geometry.iterative_projection_parameters import IterativeProjectionParameters
from jet_leg_common.jet_leg.computational_geometry.computational_geometry import ComputationalGeometry
from jet_leg_common.jet_leg.optimization.lp_vertex_redundancy import LpVertexRedundnacy

import matplotlib.pyplot as plt
from jet_leg_common.jet_leg.plotting.arrow3D import Arrow3D
        
plt.close('all')
math = Math()

''' Set the robot's name (either 'hyq', 'hyqreal', 'anymal_boxy' or 'anymal_coyote')'''
robot_name = 'anymal_coyote'

''' number of generators, i.e. rays/edges used to linearize the friction cone '''
ng = 4

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

# number of decision variables of the problem
#n = nc*6
comWF = np.array([-0.000, 0.0, 1.000])  # pos of COM in world frame w. trunk controller
# comBF = np.array([-0.0094, 0.0002, -0.0458])  # pos of COM in body frame w. trunk controller
rpy_base = np.array([0.0, 0.0, 0.0])  # orientation of body frame w. trunk controller
comWF_lin_acc = np.array([.0, .0, .0])
comWF_ang_acc = np.array([.0, .0, .0])
comWF_lin_vel = np.array([0.0, 0.0, 0.0])

""" contact points in the Base Frame"""
LF_foot = np.array([ 0.39,  0.32, -0.47])  # Starting configuration w.o. trunk controller
RF_foot = np.array([0.39, -0.32, -0.47])
LH_foot = np.array([-0.39,  0.32, -0.47])
RH_foot = np.array([-0.39, -0.32, -0.47])

contactsBF = np.vstack((LF_foot, RF_foot, LH_foot, RH_foot))
contactsWF = copy.copy(contactsBF);
rot = Rot.from_euler('xyz', [rpy_base[0], rpy_base[1], rpy_base[2]], degrees=False)
W_R_B = rot.as_dcm()
for j in np.arange(0,4):
    contactsWF[j,:] = np.add(np.dot(W_R_B,copy.copy(contactsBF[j, :])), comWF)

''' parameters to be tuned'''
mu = 0.7

''' stanceFeet vector contains 1 is the foot is on the ground and 0 if it is in the air'''
stanceFeet = [1,1,1,1]

randomSwingLeg = random.randint(0,3)
tripleStance = False # if you want you can define a swing leg using this variable
if tripleStance:
    print('Swing leg', randomSwingLeg)
    stanceFeet[randomSwingLeg] = 0
print('stanceLegs ' ,stanceFeet)

''' now I define the normals to the surface of the contact points. By default they are all vertical now'''
axisZ= array([[0.0], [0.0], [1.0]])

n1 = np.transpose(np.transpose(math.rpyToRot(0.0,0.0,0.0)).dot(axisZ))  # LF
n2 = np.transpose(np.transpose(math.rpyToRot(0.0,0.0,0.0)).dot(axisZ))  # RF
n3 = np.transpose(np.transpose(math.rpyToRot(0.0,0.0,0.0)).dot(axisZ))  # LH
n4 = np.transpose(np.transpose(math.rpyToRot(0.0,0.0,0.0)).dot(axisZ))  # RH
normals = np.vstack([n1, n2, n3, n4])

''' extForceW is an optional external pure force (no external torque for now) applied on the CoM of the robot.'''
extForceW = np.array([-000.0, 0.0, 0.0*9.81]) # units are Nm
extTorqueW = np.array([000.0, -000.0, 000.0]) # units are Nm

comp_dyn = ComputationalDynamics(robot_name)

'''You now need to fill the 'params' object with all the relevant 
    informations needed for the computation of the IP'''
params = IterativeProjectionParameters(robot_name)

#params.useInstantaneousCapturePoint = True
params.setContactsPosWF(contactsWF)
params.setCoMPosWF(comWF)
# params.setCoMPosBF(comBF)
params.setOrientation(rpy_base)
params.setTorqueLims(comp_dyn.robotModel.robotModel.joint_torque_limits)
params.setActiveContacts(stanceFeet)
params.setConstraintModes(constraint_mode_IP)
params.setContactNormals(normals)
params.setFrictionCoefficient(mu)
params.setNumberOfFrictionConesEdges(4)
params.setTotalMass(comp_dyn.robotModel.robotModel.trunkMass)
params.externalForce = extForceW  # params.externalForceWF is actually used anywhere at the moment
params.externalCentroidalTorque = extTorqueW
params.setCoMLinAcc(comWF_lin_acc)
params.setCoMAngAcc(comWF_ang_acc)
params.setCoMLinVel(comWF_lin_vel)
# params.setCoMAngVel()

# print("feet_position:", contactsBF)
# print("feet_position:", contactsWF)
# print("com:", comWF)
# print("comLinVel:", params.comLinVel)
# print("com_lin_acc:", comWF_lin_acc)
# print("com_ang_acc:", comWF_ang_acc)
# print("comp_dyn.robotModel.robotModel.joint_torque_limits:", comp_dyn.robotModel.robotModel.joint_torque_limits)
# print("stance_feet:", stanceFeet)
# print("constraint_mode_ip:", constraint_mode_IP)
# print("normals:", normals)
# print("mu:", mu)
# print("comp_dyn.robotModel.robotModel.trunkMass:", comp_dyn.robotModel.robotModel.trunkMass)
''' compute iterative projection 
Outputs of "iterative_projection_bretl" are:
IP_points = resulting 2D vertices
actuation_polygons = these are the vertices of the 3D force polytopes (one per leg)
computation_time = how long it took to compute the iterative projection
'''
IP_points, force_polytopes, IP_computation_time = comp_dyn.iterative_projection_bretl(params)
print("IP_points: ", IP_points)

#print "Inequalities", comp_dyn.ineq
#print "actuation polygons"
#print actuation_polygons

'''I now check whether the given CoM configuration is stable or not'''
isCoMStable, contactForces, forcePolytopes = comp_dyn.check_equilibrium(params)
# print("isCoMStable: ", isCoMStable)
# print('contact forces', contactForces)
comp_geom = ComputationalGeometry()
facets = comp_geom.compute_halfspaces_convex_hull(IP_points[:,:2])
point2check = comp_dyn.getReferencePoint(params, "ZMP")
isPointFeasible, margin = comp_geom.isPointRedundant(facets, point2check)
print("isPointFeasible: ", isPointFeasible)
print("Margin is: ", margin)

''' compute Instantaneous Capture Point (ICP) and check if it belongs to the feasible region '''
if params.useInstantaneousCapturePoint:
    ICP = InstantaneousCapturePoint()
    icp = ICP.compute(params)
    # icp = np.append(icp,comWF[2])
    params.instantaneousCapturePoint = icp
    lpCheck = LpVertexRedundnacy()
    isIcpInsideFeasibleRegion, lambdas = lpCheck.isPointRedundant(np.transpose(IP_points), icp)

'''Plotting the contact points in the 3D figure'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_xlim(comWF[0]-0.5,comWF[0]+0.5)
ax.set_ylim(comWF[1]-0.5,comWF[1]+0.5)
ax.set_zlim(comWF[2]-0.8,comWF[2]+0.2)

nc = np.sum(stanceFeet)
stanceID = params.getStanceIndex(stanceFeet)
force_scaling_factor = 2500
#plt.plot(contacts[0:nc,0],contacts[0:nc,1],'ko',markersize=15)
fz_tot = 0.0
shoulder_position_WF = np.zeros((4,3))

for j in range(0,nc): # this will only show the contact positions and normals of the feet that are defined to be in stance
    idx = int(stanceID[j])
    ax.scatter(contactsWF[idx,0], contactsWF[idx,1], contactsWF[idx,2],c='b',s=100)
    '''CoM will be plotted in green if it is stable (i.e., if it is inside the feasible region'''
    if isPointFeasible:
        ax.scatter(comWF[0], comWF[1], comWF[2],c='g',s=100)
    else:
        ax.scatter(comWF[0], comWF[1], comWF[2],c='r',s=100)

    ''' draw 3D arrows corresponding to contact normals'''
    a = Arrow3D([contactsWF[idx,0], contactsWF[idx,0]+normals[idx,0]/10], [contactsWF[idx,1], contactsWF[idx,1]+normals[idx,1]/10],[contactsWF[idx,2], contactsWF[idx,2]+normals[idx,2]/10], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")

    ''' The black spheres represent the projection of the contact points on the same plane of the feasible region'''
    shoulder_position_BF = [contactsBF[idx,0],contactsBF[idx,1],0.0]
    rpy = params.getOrientation()
    shoulder_position_WF[j,:] = W_R_B.dot(shoulder_position_BF) + comWF
    ax.scatter(shoulder_position_WF[j,0], shoulder_position_WF[j,1], shoulder_position_WF[j,2], c='k', s=100)
    ax.add_artist(a)

print('sum of vertical forces is', fz_tot)

''' plotting Iterative Projection points '''
plotter = Plotter()
for j in range(0,nc): # this will only show the force polytopes of the feet that are defined to be in stance
    idx = int(stanceID[j])
    plotter.plot_polygon(np.transpose(IP_points))
    if (constraint_mode_IP[idx] == 'ONLY_ACTUATION') or (constraint_mode_IP[idx] == 'FRICTION_AND_ACTUATION'):
        # print("IDX poly", force_polytopes)
        plotter.plot_actuation_polygon(ax, force_polytopes.getVertices()[idx], contactsWF[idx,:], force_scaling_factor)

base_polygon = np.vstack([shoulder_position_WF, shoulder_position_WF[0,:]])
ax.plot(base_polygon[:,0],base_polygon[:,1],base_polygon[:,2], '--k')

''' 2D figure '''
fig = plt.figure()
for j in range(0,nc): # this will only show the contact positions and normals of the feet that are defined to be in stance
    idx = int(stanceID[j])
    ''' The black spheres represent the projection of the contact points on the same plane of the feasible region'''
    # I think it should show contacts in WF instead
    shoulder_position_BF = [contactsBF[idx,0],contactsBF[idx,1],0.0]
    shoulder_position_WF[j,:] = np.dot(W_R_B,shoulder_position_BF) + comWF
    h1 = plt.plot(shoulder_position_WF[j,0],shoulder_position_WF[j,1],'ko',markersize=15, label='stance feet')

h2 = plotter.plot_polygon(np.transpose(IP_points), '--b', 5, 'Feasible Region')

'''CoM will be plotted in green if it is stable (i.e., if it is inside the feasible region)'''
if isPointFeasible:
    plt.plot(point2check[0],point2check[1],'go',markersize=15, label='ground reference point')
else:
    plt.plot(point2check[0],point2check[1],'ro',markersize=15, label='ground reference point')

plt.grid()
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()
plt.show()