import numpy as np
from jet_leg_common.jet_leg.optimization import nonlinear_projection
from jet_leg.optimization import reachable_margins
from jet_leg_common.jet_leg.computational_geometry.computational_geometry import ComputationalGeometry
from jet_leg_common.jet_leg.dynamics.computational_dynamics import ComputationalDynamics
import copy
from shapely.geometry import Polygon, Point

class KinematicJacobiansWithSplitMargins:
    def __init__(self,robot_name):
        self.projection = nonlinear_projection.NonlinearProjectionBretl(robot_name)
        self.reachable_margins_xy = reachable_margins.ReachableMargin(robot_name)
        self.compGeom = ComputationalGeometry()
        self.compDyn = ComputationalDynamics(robot_name)
        self.delta = 0.001

    
    def computeComPosJacobian(self, params, contactsWF):
        jacobian = np.zeros(3)
        default = copy.deepcopy(contactsWF)

        for j in np.arange(0, 3):
            feet_pos_dim = copy.deepcopy(default[:, j])
            feet_pos_dim -= self.delta / 2.0
            feet_pos_WF_plus = copy.deepcopy(default)
            feet_pos_WF_plus[:, j] = feet_pos_dim
            isPointFeasible_fr, margin_fr1 = self.computeComMargin(params, feet_pos_WF_plus)
            isPointFeasible, margin_kr1 = self.computeCoMKinematicMargin(params, feet_pos_WF_plus)
            margin1 = min(margin_fr1, margin_kr1)
            params.setContactsPosWF(default)
            feet_pos_dim = copy.deepcopy(default[:, j])
            feet_pos_dim += self.delta / 2.0
            feet_pos_WF_minus = copy.deepcopy(default)
            feet_pos_WF_minus[:, j] = feet_pos_dim
            isPointFeasible_fr, margin_fr2 = self.computeComMargin(params, feet_pos_WF_minus)
            isPointFeasible, margin_kr2 = self.computeCoMKinematicMargin(params, feet_pos_WF_minus)
            margin2 = min(margin_fr2, margin_kr2)
            diff = margin1 - margin2
            jacobian[j] = diff / (1.0*self.delta)

        return jacobian
    
    def computeBaseOrientationJacobian(self, params):
        jacobian = np.zeros(3)
        initialBaseOrient = copy.copy(params.getOrientation())
        #print "initialBaseOrient ", initialBaseOrient

        for j in np.arange(0, 3):
            params.setOrientation(initialBaseOrient)
            params.eulerAngles[j] = initialBaseOrient[j] + self.delta / 2.0
            isPointFeasible, margin1 = self.compDyn.compute_IP_margin(params, "COM")
            params.eulerAngles = initialBaseOrient
            params.eulerAngles[j] = initialBaseOrient[j] - self.delta / 2.0
            isPointFeasible, margin2 = self.compDyn.compute_IP_margin(params, "COM")
            diff = margin1 - margin2
            jacobian[j] = diff / self.delta
            #print "[computeBaseOrientationJacobian]", margin1, margin2, self.delta, jacobian[j]

        return jacobian

    def computeComEulerAnglesJacobian(self, params):
        jacobian = np.zeros(3)

        for j in np.arange(0, 3):
            initiaPoint = copy.copy(params.getOrientation())
            eulerAngles = initiaPoint
            eulerAngles[j] = initiaPoint[j] + self.delta / 2.0
            params.setEulerAngles(eulerAngles)
            isPointFeasible, margin_fr1 = self.compDyn.compute_IP_margin(params, "COM")
            com_check = params.getCoMPosWF()
            polygon, computation_time = self.projection.project_polytope(params, com_check, 10. * np.pi / 180, 0.001)
            margin_kr1 = 0
            if polygon.size > 0 and np.any(polygon):
                sPolygon = Polygon(polygon[:-1,:2])
                sPoint = Point(com_check[:2])
                margin_kr1 = sPoint.distance(sPolygon.exterior)
                isPointFeasible = True
            margin1 = min(margin_fr1, margin_kr1)

            initiaPoint = copy.copy(params.getOrientation())
            eulerAngles = initiaPoint
            eulerAngles[j] = initiaPoint[j] - self.delta / 2.0
            params.setEulerAngles(eulerAngles)
            isPointFeasible, margin_fr2 = self.compDyn.compute_IP_margin(params, "COM")
            com_check = params.getCoMPosWF()
            polygon, computation_time = self.projection.project_polytope(params, com_check, 10. * np.pi / 180, 0.001)
            margin_kr2 = 0
            if polygon.size > 0 and np.any(polygon):
                sPolygon = Polygon(polygon[:-1,:2])
                sPoint = Point(com_check[:2])
                margin_kr2 = sPoint.distance(sPolygon.exterior)
                isPointFeasible = True
                print("dist", margin_kr2)
            margin2 = min(margin_fr2, margin_kr2)

            diff = margin1 - margin2
            jacobian[j] = diff / self.delta

        return jacobian
    
    def computeComPosJacobianSplitMargins(self, params, contactsWF):
        jacobians = [np.zeros(3), np.zeros(3)]
        default = copy.deepcopy(contactsWF)
        for j in np.arange(0, 3):
            feet_pos_dim = copy.deepcopy(default[:, j])
            feet_pos_dim -= self.delta / 2.0
            feet_pos_WF_plus = copy.deepcopy(default)
            feet_pos_WF_plus[:, j] = feet_pos_dim
            params.setContactsPosWF(feet_pos_WF_plus)
            margins_kr1 = 0
            feasible, margins_kr1 = self.computeKinematicMarginsSplit(params)
            
            feet_pos_dim = copy.deepcopy(default[:, j])
            feet_pos_dim += self.delta / 2.0
            feet_pos_WF_minus = copy.deepcopy(default)
            feet_pos_WF_minus[:, j] = feet_pos_dim
            params.setContactsPosWF(feet_pos_WF_minus)
            margins_kr2 = 0
            feasible, margins_kr2 = self.computeKinematicMarginsSplit(params)
            
            for i in np.arange(0, 2):
                diff = margins_kr1[i] - margins_kr2[i]
                jacobians[i][j] = diff / self.delta

        return jacobians

    def computeComEulerAnglesJacobianSplitMargins(self, params):
        jacobians = [np.zeros(3), np.zeros(3)]

        for j in np.arange(0, 3):
            initiaPoint = copy.copy(params.getOrientation())
            eulerAngles = initiaPoint
            eulerAngles[j] = initiaPoint[j] + self.delta / 2.0
            params.setEulerAngles(eulerAngles)
            margins_kr1 = 0
            feasible, margins_kr1 = self.computeKinematicMarginsSplit(params)

            initiaPoint = copy.copy(params.getOrientation())
            eulerAngles = initiaPoint
            eulerAngles[j] = initiaPoint[j] - self.delta / 2.0
            params.setEulerAngles(eulerAngles)
            margins_kr2 = 0
            feasible, margins_kr2 = self.computeKinematicMarginsSplit(params)

            for i in np.arange(0, 2):
                diff = margins_kr1[i] - margins_kr2[i]
                jacobians[i][j] = diff / self.delta

        return jacobians

    def computeComLinVelJacobian(self, params):
        jacobian = np.zeros(3)
        initiaPoint = params.comLinVel

        for j in np.arange(0, 3):
            params.comLinVel = initiaPoint
            params.comLinVel[j] = initiaPoint[j] + self.delta / 2.0
            isPointFeasible, margin1 = self.compDyn.compute_IP_margin(params, "COM")

            params.comLinVel = initiaPoint
            params.comLinVel[j] = initiaPoint[j] - self.delta / 2.0
            isPointFeasible, margin2 = self.compDyn.compute_IP_margin(params, "COM")
            diff = margin1 - margin2
            jacobian[j] = diff / self.delta

        return jacobian

    def computeComLinAccJacobian(self, params):
        jacobian = np.zeros(3)
        initiaPoint = params.comLinAcc

        for j in np.arange(0, 3):
            params.comLinAcc = initiaPoint
            params.comLinAcc[j] = initiaPoint[j] + self.delta / 2.0
            isPointFeasible, margin1 = self.compDyn.compute_IP_margin(params, "COM")

            params.comLinAcc = initiaPoint
            params.comLinAcc[j] = initiaPoint[j] - self.delta / 2.0
            isPointFeasible, margin2 = self.compDyn.compute_IP_margin(params, "COM")
            diff = margin1 - margin2
            jacobian[j] = diff / self.delta

        return jacobian
    
    def computeFootJacobian(self, params, footID):
        jacobian = np.zeros(3)
        initialContacts = copy.deepcopy(params.getContactsPosWF())
        initiaPoint = copy.deepcopy(initialContacts[footID])
        for j in np.arange(0, 3):
            contacts = copy.deepcopy(initialContacts)
            print("contacts", contacts)
            contacts[footID,j] = initiaPoint[j] + self.delta / 2.0
            print("contacts1", contacts)
            isPointFeasible_fr, margin_fr1 = self.computeComMargin(params, contacts)
            isPointFeasible, margin_kr1 = self.computeCoMKinematicMargin(params, contacts)
            margin1 = min(margin_fr1, margin_kr1)
            print("margin1", margin1)
            contacts = copy.deepcopy(initialContacts)
            contacts[footID, j] = initiaPoint[j] - self.delta / 2.0
            print("contacts2", contacts)
            isPointFeasible_fr, margin_fr2 = self.computeComMargin(params, contacts)
            isPointFeasible, margin_kr2 = self.computeCoMKinematicMargin(params, contacts)
            margin2 = min(margin_fr2, margin_kr2)
            print("margin2", margin2)
            diff = margin1 - margin2
            jacobian[j] = diff / self.delta

        return jacobian
    
    def computeForceJacobian(self, params, margin_kr):
        jacobian = np.zeros(3)
        initialPoint = copy.deepcopy(params.getExternalForce())

        for j in np.arange(0, 3):
            params.externalForce = copy.deepcopy(initialPoint)
            params.externalForce[j] = initialPoint[j] + self.delta / 2.0
            isPointFeasible, margin_fr1 = self.compDyn.compute_IP_margin(params, "COM")
            margin1 = min(margin_fr1, margin_kr)
            params.externalForce = copy.deepcopy(initialPoint)
            params.externalForce[j] = initialPoint[j] - self.delta / 2.0
            isPointFeasible, margin_fr2 = self.compDyn.compute_IP_margin(params, "COM")
            margin2 = min(margin_fr2, margin_kr)
            diff = margin1 - margin2
            jacobian[j] = diff / (1.0*self.delta)

        return jacobian

    
    def plotMarginAndJacobianOfMarginWrtComLinAcceleration(self, params, acc_range, dimension):
        num_of_tests = np.shape(acc_range)
        margin = np.zeros(num_of_tests)
        jac_com_lin_acc = np.zeros((3,num_of_tests[0]))
        count = 0
        for delta_acc in acc_range:
            # print("delta_acc", delta_acc)
            params.comLinAcc = [0.0, 0.0, 0.0]
            params.comLinAcc[dimension] = delta_acc
            # print("count", count, params.comLinAcc, params.comPositionWF, params.getContactsPosWF())
            ''' compute iterative projection 
            Outputs of "iterative_projection_bretl" are:
            IP_points = resulting 2D vertices
            actuation_polygons = these are the vertices of the 3D force polytopes (one per leg)
            computation_time = how long it took to compute the iterative projection
            '''
            IP_points, force_polytopes, IP_computation_time = self.compDyn.try_iterative_projection_bretl(params)
    
            '''I now check whether the given CoM configuration is stable or not'''
            isCoMStable, contactForces, forcePolytopes = self.compDyn.check_equilibrium(params)
            # print("is CoM stable?", isCoMStable)
            #print 'Contact forces:', contactForces
            # print('IP_points:', IP_points)

            #if IP_points is True:
            #    print "ip points shape", np.shape(IP_points)
            #    facets = self.compGeom.compute_halfspaces_convex_hull(IP_points)
            #    point2check = self.compDyn.getReferencePoint(params)
            #    print "point 2 check", point2check
            #    isPointFeasible, margin[count] = self.compGeom.isPointRedundant(facets, point2check)
            isPointFeasible, margin[count] = self.compDyn.compute_IP_margin(params, "COM")
            #
            marginJAcWrtComLinAcc = self.computeComLinAccJacobian(params)
            #    print "marginJAcWrtComLinAcc", marginJAcWrtComLinAcc
            jac_com_lin_acc[:, count] = marginJAcWrtComLinAcc
            #else:
            #    jac_com_lin_acc[:, count] = [-1000] * 3

            # print "com lin vel jacobian", jac_com_lin_vel
            count += 1
    
        return margin, jac_com_lin_acc
    
    def plotMarginAndJacobianOfMarginWrtComVelocity(self, params, velocity_range, dimension):
        num_of_tests = np.shape(velocity_range)
        margin = np.zeros(num_of_tests)
        jac_com_lin_vel = np.zeros((3,num_of_tests[0]))
        count = 0
        for delta_vel in velocity_range:
            params.comLinVel = [0.0, 0.0, 0.0]
            params.comLinVel[dimension] = delta_vel
            print("count", count, params.comLinVel, params.comPositionWF, params.getContactsPosWF())
            ''' compute iterative projection 
            Outputs of "iterative_projection_bretl" are:
            IP_points = resulting 2D vertices
            actuation_polygons = these are the vertices of the 3D force polytopes (one per leg)
            computation_time = how long it took to compute the iterative projection
            '''
            #IP_points, force_polytopes, IP_computation_time = self.compDyn.iterative_projection_bretl(params)
            #
            #'''I now check whether the given CoM configuration is stable or not'''
            #isCoMStable, contactForces, forcePolytopes = self.compDyn.check_equilibrium(params)
            #print "is CoM stable?", isCoMStable
            ## print 'Contact forces:', contactForces
            #
            #facets = self.compGeom.compute_halfspaces_convex_hull(IP_points)
            #point2check = self.compDyn.getReferencePoint(params)
            #print "point 2 check", point2check
            #isPointFeasible, margin[count] = self.compGeom.isPointRedundant(facets, point2check)
            isPointFeasible, margin[count] = self.compDyn.compute_IP_margin(params, "COM")
            # print("margin:", margin[count])

            marginJAcWrtComLinVel = self.computeComLinVelJacobian(params)
            # print("marginJAcWrtComLinVel", marginJAcWrtComLinVel)
            jac_com_lin_vel[:,count] = marginJAcWrtComLinVel
            # print "com lin vel jacobian", jac_com_lin_vel

            count += 1

        return margin, jac_com_lin_vel
    
    # def computeComMargin(self, params, new_contacts_wf):
    #     default_feet_pos_WF = copy.deepcopy(params.getContactsPosWF())
    #     params.setContactsPosWF(new_contacts_wf)
    #     isPointFeasible, margin = self.compDyn.compute_IP_margin(params, "COM")
    #     params.setContactsPosWF(default_feet_pos_WF)
    #     return isPointFeasible, margin
    
    def computeComMargin(self, params, new_contacts_wf):
        default_feet_pos_WF = copy.deepcopy(params.getContactsPosWF())
        params.setContactsPosWF(new_contacts_wf)
        isPointFeasible, margin = self.compDyn.compute_IP_margin(params, "COM")
        params.setContactsPosWF(default_feet_pos_WF)
        return isPointFeasible, margin

    def computeCoMKinematicMargin(self, params, new_contacts_wf, reference_type = "COM"):

        com_check = params.getCoMPosWF()
        params.setContactsPosWF(new_contacts_wf)
        polygon, computation_time = self.projection.project_polytope(params, com_check, 10. * np.pi / 180, 0.001)
        if polygon.size > 0 and np.any(polygon):
            sPolygon = Polygon(polygon[:-1,:2])
            sPoint = Point(com_check[:2])
            dist = sPoint.distance(sPolygon.exterior)
            isPointFeasible = True
            if not sPolygon.contains(sPoint):
                isPointFeasible = False
                dist = -1 * dist
            return isPointFeasible, dist
        else:
            print("Warning! IP failed.")
            return False, -1.0
    
    def computeKinematicMargin(self, params):

        com_check = params.getCoMPosWF()
        polygon, computation_time = self.projection.project_polytope(params, com_check, 10. * np.pi / 180, 0.001)    
        if polygon.size > 0 and np.any(polygon):
            sPolygon = Polygon(polygon[:-1,:2])
            sPoint = Point(com_check[:2])
            dist = sPoint.distance(sPolygon.exterior)
            isPointFeasible = True
            if not sPolygon.contains(sPoint):
                isPointFeasible = False
                dist = -1 * dist
            return isPointFeasible, dist
        else:
            print("Warning! IP failed.")
            return False, -1.0

    def computeKinematicMarginsSplit(self, params):

        com_check = params.getCoMPosWF()
        margins, computation_time = self.reachable_margins_xy.compute_reachable_margins(params, com_check, 2, 0.001)
        if margins.size > 0 and np.any(margins):
            return True, margins
        else:
            print("Warning! IP failed.")
            return False, np.array([-1.0, -1.0])

    def plotMarginAndJacobianWrtComPosition(self, params, com_pos_range, dim_to_check):
        num_of_tests = np.shape(com_pos_range)
        margin = np.zeros(num_of_tests)
        margins_split = np.zeros((2, num_of_tests[0]))
        jac_com_pos = np.zeros((3,num_of_tests[0]))
        jac_com_pos_split = np.zeros((2, 3, num_of_tests[0]))
        count = 0
        default_feet_pos_WF = copy.deepcopy(params.getContactsPosWF())

        for delta in com_pos_range:
            """ contact points in the World Frame"""
            contactsWF = copy.deepcopy(default_feet_pos_WF) 
            contactsWF[:, dim_to_check] -= delta
            contactsWF_copy = copy.deepcopy(contactsWF) 
            isPointFeasible_fr, margin_fr = self.computeComMargin(params, contactsWF)
            isPointFeasible_kin, margin_kin = self.computeCoMKinematicMargin(params, contactsWF)
            margin[count] = min(margin_fr, margin_kin)
            jac_com_pos[:, count] = self.computeComPosJacobian(params, contactsWF)
            params.setContactsPosWF(contactsWF_copy)

            isPointFeasible, margin_split = self.computeKinematicMarginsSplit(params)
            margins_split[:, count] = margin_split
            marginJAcWrtCoMPosSPLIT = self.computeComPosJacobianSplitMargins(params, contactsWF_copy)
            jac_com_pos_split[:, :, count] = marginJAcWrtCoMPosSPLIT[:]

            count += 1
        params.setContactsPosWF(default_feet_pos_WF)
        return margin, margins_split, jac_com_pos, jac_com_pos_split

    '''
    @brief this function computes the jacobian of the feasible region margin wrt to the base orientation 
    for a predefined set of base orienation values
    @params params = iterative projection parameters 
    @params dim_to_check = 0 (roll), dim_to_check = 1 (pitch)
    @params base_orient_range = set of values to use for the roll/pitch test
    '''
    def plotMarginAndJacobianWrtBaseOrientation(self, params, base_orient_range, dim_to_check):
        num_of_tests = np.shape(base_orient_range)
        margin = np.zeros(num_of_tests)
        margins_split = np.zeros((2, num_of_tests[0]))
        jac_base_orient = np.zeros((3, num_of_tests[0]))
        jac_base_orient_split = np.zeros((2, 3, num_of_tests[0]))
        count = 0
        default_euler_angles = copy.deepcopy(params.getOrientation())

        for delta in base_orient_range:
            eulerAngles = copy.deepcopy(default_euler_angles)
            eulerAngles[dim_to_check] += delta
            eulerAngles_copy = copy.deepcopy(eulerAngles)
            params.setOrientation(eulerAngles)

            isPointFeasible, margin_fr = self.compDyn.compute_IP_margin(params, "COM")
            print("params.eurlerAngles ", params.getOrientation())
            isPointFeasible_kin, margin_kin = self.computeKinematicMargin(params)
            margin[count] = min(margin_fr, margin_kin)
            marginJAcWrtBaseOrient = self.computeComEulerAnglesJacobian(params)
            jac_base_orient[:, count] = marginJAcWrtBaseOrient
            params.setOrientation(eulerAngles_copy)

            isPointFeasible, margin_split = self.computeKinematicMarginsSplit(params)
            margins_split[:, count] = margin_split
            marginJAcWrtBaseOrientSPLIT = self.computeComEulerAnglesJacobianSplitMargins(params)
            jac_base_orient_split[:, :, count] = marginJAcWrtBaseOrientSPLIT[:]
            
            count += 1
            
        params.setOrientation(default_euler_angles)

        return margin, margins_split, jac_base_orient, jac_base_orient_split

    def plotMarginAndJacobianWrtFootPosition(self, params, foot_id, foot_pos_range, dimension_to_check):
        num_of_tests = np.shape(foot_pos_range)
        margin = np.zeros(num_of_tests)
        jac_foot_pos = np.zeros(num_of_tests)
        count = 0
        default_feet_pos_WF = copy.deepcopy(params.getContactsPosWF())

        for delta in foot_pos_range:
            # """ contact points in the World Frame"""
            # contactsWF = np.vstack((LF_foot, RF_foot, LH_foot, RH_foot))
            contactsWF = copy.deepcopy(default_feet_pos_WF)
            contactsWF[foot_id, dimension_to_check] += delta
            # params.setContactsPosWF(contactsWF)
            isPointFeasible_fr, margin_fr = self.computeComMargin(params, contactsWF)
            print("margin_fr", margin_fr)
            isPointFeasible_kin, margin_kin = self.computeCoMKinematicMargin(params, contactsWF)
            print("margin_kin", margin_kin)
            margin[count] = min(margin_fr, margin_kin)
            params.setContactsPosWF(contactsWF)
            marginJacWrtFootPos = self.computeFootJacobian(params, foot_id)
            jac_foot_pos[count] = marginJacWrtFootPos[dimension_to_check]
            params.setContactsPosWF(default_feet_pos_WF)
            count += 1

        return margin, jac_foot_pos
    
    def plotMarginAndJacobianWrtForce(self, params, force_range, dim_to_check):
        num_of_tests = np.shape(force_range)
        margin = np.zeros(num_of_tests)
        jac_force = np.zeros((3,num_of_tests[0]))
        count = 0
        default_force_WF = copy.deepcopy(params.getExternalForce())
        isPointFeasible_kin, margin_kin = self.computeKinematicMargin(params)

        for delta in force_range:
            """ contact points in the World Frame"""
            extForceWF = copy.deepcopy(default_force_WF)
            extForceWF[dim_to_check] += delta
            params.externalForce = extForceWF
            isPointFeasible, margin_fr = self.compDyn.compute_IP_margin(params, "COM")
            margin[count] = min(margin_fr, margin_kin)
            marginJacWrtForce = self.computeForceJacobian(params, margin_kin)
            jac_force[:, count] = marginJacWrtForce

            count += 1
        params.externalForce = default_force_WF

        return margin, jac_force