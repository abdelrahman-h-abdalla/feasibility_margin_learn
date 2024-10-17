import matplotlib.pyplot as plt
import numpy as np
import time
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from scipy.optimize import approx_fprime

import torch
import torch.nn.functional as F

from common.paths import ProjectPaths
from common.datasets import TrainingDataset
from common.networks import MultiLayerPerceptron

from jet_leg_common.jet_leg.computational_geometry.math_tools import Math
from jet_leg_common.jet_leg.optimization import nonlinear_projection
from jet_leg_common.jet_leg.dynamics.computational_dynamics import ComputationalDynamics
from jet_leg_common.jet_leg.computational_geometry.computational_geometry import ComputationalGeometry
from jet_leg_common.jet_leg.computational_geometry.iterative_projection_parameters import IterativeProjectionParameters
from jet_leg_common.jet_leg.optimization.jacobians_kinematics import KinematicJacobians


robot_name = 'hyqreal'

num_of_tests = 150
delta_pos_range_x = 0.8
delta_pos_range_y = 0.8
delta_pos_range_z = 0.8
delta_orient_range_vec_x = np.linspace(-delta_pos_range_x/2.0, delta_pos_range_x/2.0, num_of_tests)
delta_orient_range_vec_y = np.linspace(-delta_pos_range_y/2.0, delta_pos_range_y/2.0, num_of_tests)
delta_orient_range_vec_z = np.linspace(-delta_pos_range_z/2.0, delta_pos_range_z/2.0, num_of_tests)
gradient_epsilon = 1e-8

def g(euler):

    g = np.array([
        np.sin(euler[0])*np.sin(euler[2]) - np.cos(euler[0])*np.cos(euler[2])*np.sin(euler[1]),
        np.cos(euler[2])*np.sin(euler[0]) + np.cos(euler[0])*np.sin(euler[1])*np.sin(euler[2]),
        np.cos(euler[0])*np.cos(euler[1])
    ])

    return g

def g_component(euler, index):
    return g(euler)[index]

def plot_learnt_margin():

    paths = ProjectPaths()
    dataset_handler = TrainingDataset('../trained_models/hyqreal/seperate(kinematic)/1111', robot_name=robot_name)
    model_directory = '/hyqreal/seperate(kinematic)/1111/'
    # dataset_handler = TrainingDataset('../trained_models/final/stability_margin', robot_name=robot_name, in_dim=15)
    # model_directory = '/final/stability_margin/'
    model_directory = paths.TRAINED_MODELS_PATH + model_directory
    print("model directory: ", model_directory)
    network = MultiLayerPerceptron(in_dim=40, out_dim=1, hidden_layers=[256,256,128], activation='relu', dropout=0.0)
    # network.load_state_dict(torch.load(model_directory + 'network_state_dict.pt'))
    network.load_params_from_txt(model_directory + 'network_parameters_hyqreal.txt')
    network.eval()

    ## Margin LF X
    out_array_x = []
    jac_array_x = []
    for delta in delta_orient_range_vec_x:

        # Compute rotation matrix (from base to world) from roll/pitch/yaw - Intrinsic rotations.
        euler_angles = np.array([delta, 0.0, 0.0])
        print("euler_angles:", euler_angles)
        rotation_matrix = R.from_euler('XYZ', euler_angles, degrees=False).as_dcm() #
        print("rotation_matrix[2]:", rotation_matrix[2])
        # print("g(euler_angles):", g(euler_angles))
        P_LF_BF = np.dot(rotation_matrix.transpose(), (np.array([0.24, 0.34, -0.5]) - np.array([0.006, -0.002, -0.044]))) + \
                    np.array([0.006, -0.002, -0.044])
        P_RF_BF = np.dot(rotation_matrix.transpose(), (np.array([0.24, -0.34, -0.5]) - np.array([0.006, -0.002, -0.044]))) + \
                    np.array([0.006, -0.002, -0.044])
        P_LH_BF = np.dot(rotation_matrix.transpose(), (np.array([-0.64, 0.34, -0.5]) - np.array([0.006, -0.002, -0.044]))) + \
                    np.array([0.006, -0.002, -0.044])
        P_RH_BF = np.dot(rotation_matrix.transpose(), (np.array([-0.64, -0.34, -0.5]) - np.array([0.006, -0.002, -0.044]))) + \
                    np.array([0.006, -0.002, -0.044])
        # g = rotation_matrix[:, 2]
        network_input = np.concatenate([
            rotation_matrix[2],
            # euler_angles,
            np.zeros(12),
            P_LF_BF,
            P_RF_BF,
            P_LH_BF,
            P_RH_BF,
            np.array([0.6]),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        network_input = dataset_handler.scale_input(network_input)
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        out_array_x.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))

        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        # jacobian_g_euler = np.zeros((3, 3))
        # for i in range(3):
        #     jacobian_g_euler[:, i] = approx_fprime(euler_angles, lambda x: g_component(x, i), gradient_epsilon)
        # print("jacobian_g_euler:", jacobian_g_euler)
        gradient_g = unnormalized_gradient[0:3]
        # rotation_matrices
        # jac_array_x.append(unnormalized_gradient[15])
        dG_dPhi = np.array([np.cos(euler_angles[0]) * np.sin(euler_angles[2]) + np.sin(euler_angles[0]) * np.cos(euler_angles[2]) * np.sin(euler_angles[1]),
                            np.cos(euler_angles[2]) * np.cos(euler_angles[0]) - np.sin(euler_angles[0]) * np.sin(euler_angles[1]) * np.sin(euler_angles[2]),
                            - np.sin(euler_angles[0]) * np.cos(euler_angles[1])])
        jac_array_x.append(gradient_g.dot(dG_dPhi))

    ## Margin LF Y
    out_array_y = []
    jac_array_y = []
    for delta in delta_orient_range_vec_y:
        # Compute rotation matrix (from base to world) from roll/pitch/yaw - Intrinsic rotations.
        euler_angles = np.array([0., delta, 0.0])
        print("euler_angles:", euler_angles)
        rotation_matrix = R.from_euler('XYZ', np.array([0.0, delta, 0.0]), degrees=False).as_dcm() #
        print("rotation_matrix[2]:", rotation_matrix[2])
        g = rotation_matrix[:, 2]
        P_LF_BF = np.dot(rotation_matrix.transpose(), (np.array([0.24, 0.34, -0.5]) - np.array([0.006, -0.002, -0.044]))) + \
                    np.array([0.006, -0.002, -0.044])
        P_RF_BF = np.dot(rotation_matrix.transpose(), (np.array([0.24, -0.34, -0.5]) - np.array([0.006, -0.002, -0.044]))) + \
                    np.array([0.006, -0.002, -0.044])
        P_LH_BF = np.dot(rotation_matrix.transpose(), (np.array([-0.64, 0.34, -0.5]) - np.array([0.006, -0.002, -0.044]))) + \
                    np.array([0.006, -0.002, -0.044])
        P_RH_BF = np.dot(rotation_matrix.transpose(), (np.array([-0.64, -0.34, -0.5]) - np.array([0.006, -0.002, -0.044]))) + \
                    np.array([0.006, -0.002, -0.044])
        network_input = np.concatenate([
            rotation_matrix[2],
            # euler_angles,
            np.zeros(12),
            P_LF_BF,
            P_RF_BF,
            P_LH_BF,
            P_RH_BF,
            np.array([0.6]),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        print("network_input Y:", network_input)
        network_input = dataset_handler.scale_input(network_input)
        print("network_input scaled Y:", network_input)
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        print("output:", output)
        out_array_y.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))
        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        gradient_g = unnormalized_gradient[0:3]
        dG_dTheta = np.array([-np.cos(euler_angles[0]) * np.cos(euler_angles[2]) * np.cos(euler_angles[1]),
                              np.cos(euler_angles[0]) * np.cos(euler_angles[1]) * np.sin(euler_angles[2]),
                              -np.cos(euler_angles[0]) * np.sin(euler_angles[1])])
        # jac_array_y.append(unnormalized_gradient[16])
        # jac_array_y.append(unnormalized_gradient[16])
        jac_array_y.append(gradient_g.dot(dG_dTheta))

    ## Margin LF Z
    out_array_z = []
    jac_array_z = []
    for delta in delta_orient_range_vec_z:
        # Compute rotation matrix (from base to world) from roll/pitch/yaw - Intrinsic rotations.
        euler_angles = np.array([0., 0.0, delta])
        rotation_matrix = R.from_euler('XYZ', np.array([0.0, 0.0, delta]), degrees=False).as_dcm() #
        # g = rotation_matrix[:, 2]
        network_input = np.concatenate([
            rotation_matrix[2],
            # euler_angles,
            np.zeros(12),
            P_LF_BF,
            P_RF_BF,
            P_LH_BF,
            P_RH_BF,
            np.array([0.6]),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        P_LF_BF = np.dot(rotation_matrix.transpose(), (np.array([0.24, 0.34, -0.5]) - np.array([0.006, -0.002, -0.044]))) + \
                    np.array([0.006, -0.002, -0.044])
        P_RF_BF = np.dot(rotation_matrix.transpose(), (np.array([0.24, -0.34, -0.5]) - np.array([0.006, -0.002, -0.044]))) + \
                    np.array([0.006, -0.002, -0.044])
        P_LH_BF = np.dot(rotation_matrix.transpose(), (np.array([-0.64, 0.34, -0.5]) - np.array([0.006, -0.002, -0.044]))) + \
                    np.array([0.006, -0.002, -0.044])
        P_RH_BF = np.dot(rotation_matrix.transpose(), (np.array([-0.64, -0.34, -0.5]) - np.array([0.006, -0.002, -0.044]))) + \
                    np.array([0.006, -0.002, -0.044])
        network_input = dataset_handler.scale_input(network_input)
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        out_array_z.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))
        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        gradient_g = unnormalized_gradient[0:3]
        # jac_array_z.append(unnormalized_gradient[17])
        # jac_array_z.append(unnormalized_gradient[17])
        dG_dPsi = np.array([np.sin(euler_angles[0]) * np.cos(euler_angles[2]) + np.cos(euler_angles[0]) * np.sin(euler_angles[2]) * np.sin(euler_angles[1]),
                            - np.sin(euler_angles[2]) * np.sin(euler_angles[0]) + np.cos(euler_angles[0]) * np.sin(euler_angles[1]) * np.cos(euler_angles[2]),
                            0.0])
        jac_array_z.append(gradient_g.dot(dG_dPsi))


    return [[out_array_x, out_array_y, out_array_z], [jac_array_x, jac_array_y, jac_array_z]]


##############################################################################################

def plot_analytical_margin():
    
    math = Math()

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

    comWF = np.array([0.006, -0.002, -0.044])
    comBF = np.array([0.006, -0.002, -0.044])
    comWF_lin_acc = np.array([.0, .0, .0])
    comWF_ang_acc = np.array([.0, .0, .0])

    ''' extForceW is an optional external pure force (no external torque for now) applied on the CoM of the robot.'''
    extForce = np.array([0., .0, 0.0 * 9.81])  # units are N
    extCentroidalTorque = np.array([.0, .0, .0])  # units are Nm
    extCentroidalWrench = np.hstack([extForce, extCentroidalTorque])

    ''' parameters to be tuned'''
    mu = 0.6

    ''' stanceFeet vector contains 1 is the foot is on the ground and 0 if it is in the air'''
    stanceFeet = [1, 1, 1, 1]


    ''' now I define the normals to the surface of the contact points. By default they are all vertical now'''
    axisZ = np.array([[0.0], [0.0], [1.0]])

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
    params = IterativeProjectionParameters(robot_name)
    """ contact points in the World Frame"""
    LF_foot = np.array([0.24, 0.34, -0.5])
    RF_foot = np.array([0.24, -0.34, -0.5])
    LH_foot = np.array([-0.64, 0.34, -0.5])
    RH_foot = np.array([-0.64, -0.34, -0.5])

    contactsWF = np.vstack((LF_foot, RF_foot, LH_foot, RH_foot))

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
    params.useInstantaneousCapturePoint = True
    params.externalCentroidalWrench = extCentroidalWrench
    params.setCoMPosWF(comWF)
    params.setCoMPosBF(comBF)
    params.comLinVel = [0., 0.0, 0.0]
    params.setCoMLinAcc(comWF_lin_acc)
    params.setTorqueLims(comp_dyn.robotModel.robotModel.joint_torque_limits)
    params.setJointLimsMax(joint_limits_max)
    params.setJointLimsMin(joint_limits_min)
    params.setActiveContacts(stanceFeet)
    params.setConstraintModes(constraint_mode_IP)
    params.setContactNormals(normals)
    params.setFrictionCoefficient(mu)
    params.setTotalMass(comp_dyn.robotModel.robotModel.trunkMass)
    params.externalForceWF = extForceW  # params.externalForceWF is actually used anywhere at the moment

    params_base_roll = deepcopy(params)
    params_base_pitch = deepcopy(params)

    jac = KinematicJacobians(robot_name)
    comp_geom = ComputationalGeometry()


    ''' Margin LF '''
    params_euler_x = deepcopy(params)
    params_euler_y = deepcopy(params)
    params_euler_z = deepcopy(params)
    foot_id = 0
    margin_euler_x, jac_euler_x = jac.plotMarginAndJacobianWrtBaseOrientation(params_euler_x, delta_orient_range_vec_x, 0)
    margin_euler_y, jac_euler_y = jac.plotMarginAndJacobianWrtBaseOrientation(params_euler_y, delta_orient_range_vec_y, 1)
    margin_euler_z, jac_euler_z = jac.plotMarginAndJacobianWrtBaseOrientation(params_euler_z, delta_orient_range_vec_z, 2)


    print("pos_margin_x:", margin_euler_x)
    print("pos_margin_y:", margin_euler_y)
    print("pos_margin_z:", margin_euler_z)
    print("jac_com_pos_x:", jac_euler_x)
    print("jac_com_pos_y:", jac_euler_y)
    print("jac_com_pos_z:", jac_euler_z)
    
        
    return [[margin_euler_x, margin_euler_y, margin_euler_z], [jac_euler_x, jac_euler_y, jac_euler_z]]

#####################################################################################################

def main():

    margin_learnt, jacobian_learnt = plot_learnt_margin()
    print("Computed learnt margin")
    margin, jacobian = plot_analytical_margin()
    print("Computed analytical margin")
    ### Plotting

    ## X axis
    fig1 = plt.figure(1)
    fig1.suptitle("Stability margin")

    plt.subplot(131)
    plt.plot(delta_orient_range_vec_x, margin[0], 'g', markersize=15, label='CoM')
    # plt.plot(delta_orient_range_vec_x, margin_learnt[0], 'b', markersize=15, label='CoM')
    plt.grid()
    plt.xlabel("$\theta_x$ [m]")
    plt.ylabel("m [m]")
    plt.title("Roll margin")

    plt.subplot(132)
    plt.plot(delta_orient_range_vec_y, margin[1], 'g', markersize=15, label='CoM')
    # plt.plot(delta_orient_range_vec_y, margin_learnt[1], 'b', markersize=15, label='CoM')
    plt.grid()
    plt.xlabel("$\theta_x$ [m]")
    plt.ylabel("m [m]")
    plt.title("Pitch margin")

    plt.subplot(133)
    plt.plot(delta_orient_range_vec_z, margin[2], 'g', markersize=15, label='CoM')
    # plt.plot(delta_orient_range_vec_z, margin_learnt[2], 'b', markersize=15, label='CoM')
    plt.grid()
    plt.xlabel("$\theta_z$ [m]")
    plt.ylabel("m [m]")
    plt.title("Yaw margin")

    ## Jacobians
    fig2 = plt.figure(2)
    fig2.suptitle("Gradients")
    print("jacobian", jacobian)
    print("jacobian_learnt:", jacobian_learnt)

    plt.subplot(131)
    plt.plot(delta_orient_range_vec_x, jacobian[0][0], 'g', markersize=15, label='numerical')
    # plt.plot(delta_orient_range_vec_x, jacobian_learnt[0], 'b', markersize=15, label='learn (backprop)')
    plt.grid()
    plt.xlabel("$theta_x$ [m]")
    plt.ylabel("\delta m/ \delta theta_{x}")
    # plt.title("CoM X pos margin")

    plt.subplot(132)
    plt.plot(delta_orient_range_vec_y, jacobian[1][1], 'g', markersize=15, label='numerical')
    # plt.plot(delta_orient_range_vec_y, jacobian_learnt[1], 'b', markersize=15, label='learnt (backprop)')
    plt.grid()
    plt.xlabel("$theta_y$ [m]")
    plt.ylabel("\delta m/ \delta theta_{y}")
    # plt.title("CoM Y pos margin")

    plt.subplot(133)
    plt.plot(delta_orient_range_vec_z, jacobian[2][2], 'g', markersize=15, label='numerical')
    # plt.plot(delta_orient_range_vec_z, jacobian_learnt[2], 'b', markersize=15, label='learnt (backprop)')
    plt.grid()
    plt.xlabel("$theta_z$ [m]")
    plt.ylabel("\delta m/ \delta theta_{z}")

    plt.show()


if __name__ == '__main__':
    main()