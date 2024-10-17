import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from copy import deepcopy

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
from jet_leg_common.jet_leg.optimization.jacobians_kinematics_xy import KinematicJacobiansXY


robot_name = 'hyqreal'

num_of_tests = 50
delta_pos_range_x = 0.4
delta_pos_range_y = 0.4
delta_pos_range_vec_xy = np.array([np.linspace(-delta_pos_range_x/2.0, delta_pos_range_x/2.0, num_of_tests),
                                   np.linspace(-delta_pos_range_y/2.0, delta_pos_range_y/2.0, num_of_tests)])

def plot_learnt_margin():

    paths = ProjectPaths()
    dataset_handler = TrainingDataset('../trained_models/hyqreal/seperate(kinematic)/1011', robot_name=robot_name, in_dim=34)
    model_directory = '/hyqreal/seperate(kinematic)/1011/'
    model_directory = paths.TRAINED_MODELS_PATH + model_directory
    print("model directory: ", model_directory)
    network = MultiLayerPerceptron(in_dim=34, out_dim=1, hidden_layers=[512,256,128], activation='relu', dropout=0.0)
    # network.load_state_dict(torch.load(model_directory + 'network_state_dict.pt'))
    network.load_params_from_txt(model_directory + 'network_parameters_hyqreal.txt')
    network.eval()

    ## Margin X
    out_array_xy = np.zeros((num_of_tests, num_of_tests))
    count_x = 0
    jac_array_x = []
    for delta_x in delta_pos_range_vec_xy[0]:
        count_y = 0
        for delta_y in delta_pos_range_vec_xy[1]:
            network_input = np.concatenate([
                np.array([0.0, 0.0, 1.0]),
                np.array([0., 0., 0.0]),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.array([0.44 + delta_x, 0.34 + delta_y, -0.55]),
                np.array([-0.44, 0.34, -0.55]),
                np.array([-0.44, -0.34, -0.55]),
                np.array([0.6]),
                np.array([0.0, 0.0, 1.0] * 3)
            ])
            network_input = dataset_handler.scale_input(network_input)
            network_input = torch.tensor(network_input, requires_grad=True)
            output = network(network_input.float())
            output = dataset_handler.scale_output(output)
            out_array_xy[count_x][count_y] = output.detach().numpy()[:][0]
            # Compute gradients of the output w.r.t. the input
            # output.backward(torch.ones_like(output))
            # unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
            # jac_array_x.append(unnormalized_gradient[15])
            count_y += 1
        count_x += 1


    return out_array_xy


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

    comWF = np.array([.0, 0.0, 0.0])
    comWF_lin_acc = np.array([-0.0, -0.0, .0])
    comWF_ang_acc = np.array([.0, .0, .0])

    ''' extForceW is an optional external pure force (no external torque for now) applied on the CoM of the robot.'''
    extForce = np.array([0., .0, 0.0 * 9.81])  # units are N
    extCentroidalTorque = np.array([.0, .0, .0])  # units are Nm
    extCentroidalWrench = np.hstack([extForce, extCentroidalTorque])

    ''' parameters to be tuned'''
    mu = 0.6

    ''' stanceFeet vector contains 1 is the foot is on the ground and 0 if it is in the air'''
    stanceFeet = [1, 0, 1, 1]


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
    LF_foot = np.array([0.44, 0.34, -0.55])
    RF_foot = np.array([0.44, -0.34, -0.55])
    LH_foot = np.array([-0.44, 0.34, -0.55])
    RH_foot = np.array([-0.44, -0.34, -0.55])

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

    jac = KinematicJacobiansXY(robot_name)


    ''' Margin LF '''
    params_foot_xy = deepcopy(params)

    foot_id = 0
    margin_foot_pos_xy = jac.plotMarginWrtFootPosition(params_foot_xy, foot_id, delta_pos_range_vec_xy)


    print("pos_margin_x:", margin_foot_pos_xy)
    # print("jac_com_pos_x:", jac_foot_pos_x)
    
        
    return margin_foot_pos_xy

#####################################################################################################

def main():

    margin_learnt = plot_learnt_margin()
    print("Computed learnt margin")
    margin = plot_analytical_margin()
    print("Computed analytical margin")
    ### Plotting
    plt.rcParams.update({'font.size': 12})
    # plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(top=0.85)
    ## X axis
    fig1 = plt.figure(1)
    # ax1 = fig1.add_subplot(111, projection='3d')
    ax1 = fig1.gca(projection='3d')
    fig1.suptitle("Feasibility margin (1011)", x=0.565, y=1.00)
    # delta_pos_range_vec_x = np.linspace(-7.delta_pos_range_x/2.0, -7.8 + delta_pos_range_x/2.0, num_of_tests)
    # delta_pos_range_vec_y = np.linspace(-5.delta_pos_range_y/2.0, -5.1 + delta_pos_range_y/2.0, num_of_tests)

    X, Y = np.meshgrid(delta_pos_range_vec_xy[0], delta_pos_range_vec_xy[1])
    print("X:", X)
    print("Y:", Y)
    # print("margin:\n", margin)

    # Plot both surfaces on the same axes
    ax1.plot_surface(X, Y, margin.T, color='g', alpha=0.7, rstride=1, cstride=1, label='numerical',linewidth=3.0)
    ax1.plot_surface(X, Y, margin_learnt.T, color='b', alpha=0.6, rstride=1, cstride=1, label='learnt',linewidth=3.0)
    ax1.set_xlabel("$p_x$ [m]")
    ax1.set_ylabel("$p_y$ [m]")
    ax1.set_zlabel("margin [m]")
    ax1.set_title("LF X")

    # plt.subplot(121)
    # plt.plot(delta_pos_range_vec_xy, margin[0], 'g', markersize=15, label='analytical',linewidth=3.0)
    # plt.plot(delta_pos_range_vec_xy, margin_learnt[0], 'b', markersize=15, label='learnt',linewidth=3.0)
    # plt.grid()
    # plt.xlabel("$p_x$ [m]")
    # plt.ylabel("margin [m]")
    # plt.title("LF X")
    

    # ## Jacobians
    # fig2 = plt.figure(2)
    # fig2.suptitle("Gradients")
    # print("jacobian", jacobian[0])
    # print("jacobian_learnt:", jacobian_learnt[0])

    # plt.subplot(121)
    # plt.plot(delta_pos_range_vec_x, jacobian[0], 'g', markersize=15, label='analytical',linewidth=3.0)
    # plt.plot(delta_pos_range_vec_x, jacobian_learnt[0], 'b', markersize=15, label='learnt',linewidth=3.0)
    # plt.grid()
    # plt.xlabel("$c_x$ [m]")
    # plt.ylabel("\delta m/ \delta c_{x}")
    # # plt.title("CoM X pos margin")

    # plt.subplot(122)
    # plt.plot(delta_pos_range_vec_y, jacobian[1], 'g', markersize=15, label='analytical',linewidth=3.0)
    # plt.plot(delta_pos_range_vec_y, jacobian_learnt[1], 'b', markersize=15, label='learnt',linewidth=3.0)
    # plt.grid()
    # plt.xlabel("$c_y$ [m]")
    # plt.ylabel("\delta m/ \delta c_{y}")
    # # plt.title("CoM Y pos margin")

    # plt.tight_layout()
    # plt.ylim(top=2.0)
    # fig1.tight_layout()
    # fig2.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()