# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:54:31 2018

@author: Romeo Orsolino
"""

import numpy as np

from numpy import array
from jet_leg_common.jet_leg.plotting.plotting_tools import Plotter
import random
from jet_leg_common.jet_leg.computational_geometry.math_tools import Math
from jet_leg_common.jet_leg.dynamics.computational_dynamics import ComputationalDynamics
from jet_leg_common.jet_leg.computational_geometry.computational_geometry import ComputationalGeometry
from jet_leg_common.jet_leg.computational_geometry.iterative_projection_parameters import IterativeProjectionParameters
from jet_leg_common.jet_leg.optimization.jacobians import Jacobians
from jet_leg_common.jet_leg.variables.com import CoM
import time

import matplotlib.pyplot as plt
from jet_leg_common.jet_leg.plotting.arrow3D import Arrow3D

plt.close('all')
math = Math()

''' Set the robot's name (either 'hyq', 'hyqreal' or 'anymal_boxy' or 'anymal_coyote')'''
robot_name = 'anymal_boxy'

'''
possible constraints for each foot:
 ONLY_ACTUATION = only joint-torque limits are enforces
 ONLY_FRICTION = only friction cone constraints are enforced
 FRICTION_AND_ACTUATION = both friction cone constraints and joint-torque limits
'''
constraint_mode_IP = ['ONLY_FRICTION',
                      'ONLY_FRICTION',
                      'ONLY_FRICTION',
                      'ONLY_FRICTION']

# number of decision variables of the problem
# n = nc*6
comWF = np.array([.0, 0.05, 0.0])
comWF_lin_acc = np.array([.0, .0, .0])
comWF_ang_acc = np.array([.0, .0, .0])

''' extForceW is an optional external pure force (no external torque for now) applied on the CoM of the robot.'''
extForce = np.array([0., .0, 0.0 * 9.81])  # units are N
extCentroidalTorque = np.array([.0, .0, .0])  # units are Nm
extCentroidalWrench = np.hstack([extForce, extCentroidalTorque])

""" contact points in the World Frame"""
LF_foot = np.array([0.3, 0.2, -0.4])
RF_foot = np.array([0.3, -0.2, -0.4])
LH_foot = np.array([-0.3, 0.2, -0.4])
RH_foot = np.array([-0.3, -0.2, -0.4])

contactsWF = np.vstack((LF_foot, RF_foot, LH_foot, RH_foot))

''' parameters to be tuned'''
mu = 0.5

''' stanceFeet vector contains 1 is the foot is on the ground and 0 if it is in the air'''
stanceFeet = [1, 1, 1, 1]

randomSwingLeg = random.randint(0, 3)
tripleStance = False  # if you want you can define a swing leg using this variable
if tripleStance:
    print 'Swing leg', randomSwingLeg
    stanceFeet[randomSwingLeg] = 0
print 'stanceLegs ', stanceFeet

''' now I define the normals to the surface of the contact points. By default they are all vertical now'''
axisZ = array([[0.0], [0.0], [1.0]])

n1 = np.transpose(np.transpose(math.rpyToRot(0.0, 0.0, 0.0)).dot(axisZ))  # LF
n2 = np.transpose(np.transpose(math.rpyToRot(0.0, 0.0, 0.0)).dot(axisZ))  # RF
n3 = np.transpose(np.transpose(math.rpyToRot(0.0, 0.0, 0.0)).dot(axisZ))  # LH
n4 = np.transpose(np.transpose(math.rpyToRot(0.0, 0.0, 0.0)).dot(axisZ))  # RH
normals = np.vstack([n1, n2, n3, n4])

''' extForceW is an optional external pure force (no external torque for now) applied on the CoM of the robot.'''
extForceW = np.array([0.0, 0.0, 0.0])  # units are Nm

comp_dyn = ComputationalDynamics(robot_name)

'''You now need to fill the 'params' object with all the relevant 
    informations needed for the computation of the IP'''
params = IterativeProjectionParameters()

start = time.time()

params.useContactTorque = False
params.useInstantaneousCapturePoint = True
params.setContactsPosWF(contactsWF)
params.externalCentroidalWrench = extCentroidalWrench
params.setCoMPosWF(comWF)
params.comLinVel = [0.25, 0.0, 0.0]
params.setCoMLinAcc(comWF_lin_acc)
params.setTorqueLims(comp_dyn.robotModel.robotModel.joint_torque_limits)
params.setActiveContacts(stanceFeet)
params.setConstraintModes(constraint_mode_IP)
params.setContactNormals(normals)
params.setFrictionCoefficient(mu)
params.setTotalMass(comp_dyn.robotModel.robotModel.trunkMass)
params.externalForceWF = extForceW  # params.externalForceWF is actually used anywhere at the moment

''' compute iterative projection 
Outputs of "iterative_projection_bretl" are:
IP_points = resulting 2D vertices
actuation_polygons = these are the vertices of the 3D force polytopes (one per leg)
computation_time = how long it took to compute the iterative projection
'''
IP_points, force_polytopes, IP_computation_time = comp_dyn.iterative_projection_bretl(params)

'''I now check whether the given CoM configuration is stable or not'''
isCoMStable, contactForces, forcePolytopes = comp_dyn.check_equilibrium(params)
print "is CoM stable?", isCoMStable
print 'Contact forces:', contactForces

comp_geom = ComputationalGeometry()
facets = comp_geom.compute_halfspaces_convex_hull(IP_points)
point2check = np.array([comWF[0], comWF[1]])
isPointFeasible, margin = comp_geom.isPointRedundant(facets, point2check)
print "Margin is: ", margin

state = np.hstack([params.getCoMPosWF(), params.comLinVel])
com = CoM(params)

jac = Jacobians("anymal_boxy")
jac_com_pos = jac.computeComPosJacobian(params)
#print "com pos jacobian", jac_com_pos

jac_com_orient = jac.computeComEulerAnglesJacobian(params)
#print "com orientation jacobian", jac_com_orient

jac_com_lin_vel = jac.computeComLinVelJacobian(params)
#print "com lin vel jacobian", jac_com_lin_vel

jac_com_lin_acc = jac.computeComLinVelJacobian(params)
#print "com lin acc jacobian", jac_com_lin_acc

jac_ang_vel = [0,0,0]

jac_feet = jac.computeFeetJacobian(params)
#print "LF foot jacobian", jac_feet

state_jacobian = np.hstack([jac_com_pos, jac_com_orient, jac_feet])
print("jacobian vector: ",state_jacobian)
print("total time", time.time() - start)