# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:54:31 2018

@author: Abdelrahman Abdalla
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy import array, cos, sin, cross, pi
from scipy.linalg import norm
from jet_leg.computational_geometry.geometry import Geometry
import time
import numpy as np

from jet_leg.computational_geometry.math_tools import Math
from jet_leg.kinematics.kinematics_interface import KinematicsInterface


class ReachableMargin:
	def __init__(self, robot_name):
		self.robotName = robot_name
		self.kin = KinematicsInterface(self.robotName)
		self.geom = Geometry()
		self.math = Math()

	# @property
	# def params(self):
	# 	return self.params
	#
	# @params.setter
	# def params(self, _params):
	# 	self.params = _params

	def getcontactsBF(self, params, comPositionWF):

		comPositionBF = params.getCoMPosBF()
		contactsWF = params.getContactsPosWF()
		contactsBF = np.zeros((4, 3))
		rpy = params.getOrientation()

		for j in np.arange(0, 4):
			j = int(j)
			contactsBF[j, :] = np.add(np.dot(self.math.rpyToRot(rpy[0], rpy[1], rpy[2]),
																(contactsWF[j, :] - comPositionWF)),
															comPositionBF)

		return contactsBF

	def compute_vertix(self, com_pos_x, com_pos_y, plane_normal, CoM_plane_z_intercept, params, q_0, theta, min_dir_step, max_iter):
		"""
		Compute vertix of projected polygon in vdir direction.

		Solves nonlinear optimization problem by iteratively testing
		over beta-spaced possible CoM positions along vdir direction

		Returns
		-------
		poly: Polygon Output polygon.
		"""

		# Search for a CoM position in direction vdir

		vdir = array([cos(theta), sin(theta)])

		c_t_xy = [com_pos_x, com_pos_y]  # CoM to be tested
		cxy_opt = c_t_xy  # Optimal CoM so far
		foot_vel = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
		stanceIndex = params.getStanceIndex(params.getStanceFeet())

		i = 0

		# Find closest feasible and non-feasible points to start bisection algorithm

		# Initial step is large to speed up initial search
		# However, region is not convex so too large of a step can be non-conservative and skip close vertices
		dir_step = 0.1
		c_t_feasible = True  # Flag for out of limits search

		while c_t_feasible and i < max_iter:

			c_t_xy += dir_step * vdir  # Iterate along direction vector by step cm
			z_coordinate = self.math.compute_z_component_of_plane(c_t_xy, plane_normal, CoM_plane_z_intercept)
			c_t = np.append(c_t_xy, z_coordinate)
			contactsBF = self.getcontactsBF(params, c_t)
			q = self.kin.inverse_kin(contactsBF, foot_vel, q_0)
			q_to_check = np.concatenate([list(q[leg*3 : leg*3+3]) for leg in stanceIndex]) # To check 3 or 4 feet stance

			# if (not self.kin.hyqreal_ik_success) or self.kin.isOutOfJointLims(q, params.getJointLimsMax(), params.getJointLimsMin()):
			if (not self.kin.hyqreal_ik_success) or \
					self.kin.isOutOfJointLims(q_to_check, params.getJointLimsMax()[stanceIndex,:],
											  params.getJointLimsMin()[stanceIndex,:]): # kin.hyqreal_ik_success is always true for Hyq
				c_t_feasible = False
			else:
				cxy_opt = [c_t[0], c_t[1]]
				dir_step += dir_step / 2
				q_0 = q

			i += 1

		# Perform bisection algorithm using two points from previous step
		# Switch direction to go back to feasible region
		dir_step = -dir_step / 2

		while abs(dir_step) >= min_dir_step and i < max_iter:

			old_c_t_feasible = c_t_feasible
			c_t_xy += dir_step * vdir
			z_coordinate = self.math.compute_z_component_of_plane(c_t_xy, plane_normal, CoM_plane_z_intercept)
			c_t = np.append(c_t_xy, z_coordinate)
			contactsBF = self.getcontactsBF(params, c_t)
			q = self.kin.inverse_kin(contactsBF, foot_vel, q_0)
			q_to_check = np.concatenate([list(q[leg * 3: leg * 3 + 3]) for leg in stanceIndex]) # To check 3 or 4 feet stance

			# If new point is on the same side (feasible or infeasible region) as last point, continue in same direction
			# if self.kin.hyqreal_ik_success and (not self.kin.isOutOfJointLims(q, params.getJointLimsMax(), params.getJointLimsMin())):
			if self.kin.hyqreal_ik_success and \
					(not self.kin.isOutOfJointLims(q_to_check, params.getJointLimsMax()[stanceIndex,:],
												   params.getJointLimsMin()[stanceIndex,:])):
				c_t_feasible = True
				cxy_opt = [c_t[0], c_t[1]]
				q_0 = q
			else:
				c_t_feasible = False

			if c_t_feasible == old_c_t_feasible:
				dir_step = dir_step/2
			else:
				dir_step = -dir_step / 2

			i += 1
		# print "new dir_step: ", dir_step

		return cxy_opt

	def compute_margin(self, com_pos_x, com_pos_y, plane_normal, CoM_plane_z_intercept, params, q_0, theta, min_dir_step, max_iter):
		
		comPosWF_0 = params.getCoMPosWF()
		
		v1 = self.compute_vertix(comPosWF_0[0], comPosWF_0[1], plane_normal, CoM_plane_z_intercept, params, q_0, theta, min_dir_step, max_iter)
		m1 = np.linalg.norm(params.getCoMPosWF()[:2] - v1)
		theta = theta + np.pi
		v2 = self.compute_vertix(comPosWF_0[0], comPosWF_0[1], plane_normal, CoM_plane_z_intercept, params, q_0, theta, min_dir_step, max_iter)
		m2 = np.linalg.norm(params.getCoMPosWF()[:2] - v2)
		margin = np.minimum(m1, m2)
		
		return margin

	def compute_margins(self, params, q_CoM, margins_N, dir_step, max_iter):
		"""
		Compute projected Polytope.

		Returns
		-------
		margins: list of arrays List of margins of the
				projected vertices.
		"""

		comPosWF_0 = params.getCoMPosWF()

		plane_normal = params.get_plane_normal()
		CoM_plane_z_intercept = params.get_CoM_plane_z_intercept()
		margins = np.zeros(margins_N)

		theta_step = np.pi / 2 / (margins_N - 1)  # Calculate step size

		for i in range(margins_N):
			
			theta = theta_step * i  # Calculate current angle
			# Compute region for the current CoM position (in world frame)
			m = self.compute_margin(comPosWF_0[0], comPosWF_0[1], plane_normal, CoM_plane_z_intercept, params, q_CoM, theta, dir_step, max_iter)
			margins[i] = m

		return margins

	def compute_reachable_margins(self, params, com_wf_check=None, margins_N=2, dir_step=0.03, max_iter=1000):
		"""
		Compute the reachable margins along margins_N directions.

		Returns
		-------
		vertices: list of arrays List of reachable margins of the.
		"""

		ip_start = time.time()

		# Check if current configuration is already feasible
		stanceIndex = params.getStanceIndex(params.getStanceFeet())
		contactsBF = self.getcontactsBF(params, params.getCoMPosWF())
		foot_vel = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

		if self.kin.isOutOfWorkSpace(contactsBF, params.getJointLimsMax(), params.getJointLimsMin(), stanceIndex, foot_vel):
			print("Couldn't compute a reachable region! Current configuration is already out of joint limits!")
			return np.array([]), (time.time() - ip_start)

		if com_wf_check is not None:

			contactsBF_check = self.getcontactsBF(params, com_wf_check)
			if self.kin.isOutOfWorkSpace(contactsBF_check, params.getJointLimsMax(), params.getJointLimsMin(), stanceIndex, foot_vel):
				print("Ouch!")
				return np.array([]), (time.time() - ip_start)

		# Compute region
		q_CoM = self.kin.get_current_q()
		margins = self.compute_margins(params, q_CoM, margins_N, dir_step, max_iter)
		if margins.size == 0:
			return np.array([]), (time.time() - ip_start)

		computation_time = (time.time() - ip_start)

		return margins, computation_time
