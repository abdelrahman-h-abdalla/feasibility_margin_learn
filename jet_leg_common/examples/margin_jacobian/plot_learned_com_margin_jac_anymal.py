import matplotlib.pyplot as plt
import numpy as np
import time
from copy import deepcopy

import torch
import torch.nn.functional as F

from common.paths import ProjectPaths
from common.datasets import TrainingDataset
from common.networks import MultiLayerPerceptron

from jet_leg_common.jet_leg.computational_geometry.math_tools import Math
from jet_leg_common.jet_leg.dynamics.computational_dynamics import ComputationalDynamics
from jet_leg_common.jet_leg.computational_geometry.computational_geometry import ComputationalGeometry
from jet_leg_common.jet_leg.computational_geometry.iterative_projection_parameters import IterativeProjectionParameters
from jet_leg_common.jet_leg.optimization.jacobians import Jacobians


robot_name = 'anymal_coyote'

num_of_tests = 25
delta_pos_range = 0.6
delta_pos_range_vec = np.linspace(-delta_pos_range/2.0, delta_pos_range/2.0, num_of_tests)
delta_acc_range = 8.0
delta_acc_range_vec = np.linspace(-delta_acc_range/2.0, delta_acc_range/2.0, num_of_tests)

def plot_learnt_margin():

    paths = ProjectPaths()
    dataset_handler = TrainingDataset()
    model_directory = '/final/stability_margin/'
    model_directory = paths.TRAINED_MODELS_PATH + model_directory
    print("model directory: ", model_directory)
    network = MultiLayerPerceptron(in_dim=47, out_dim=1, hidden_layers=[256,128,128], activation=F.softsign, dropout=0.0)
    # network.load_state_dict(torch.load(model_directory + 'network_state_dict.pt'))
    network.load_params_from_txt(model_directory + 'network_parameters_anymal_c.txt')
    network.eval()

    ## Margin X
    out_array_x = []
    jac_array_x = []
    for delta in delta_pos_range_vec:
        network_input = np.concatenate([
            np.array([0.0, 0.0, 1.0]),
            np.zeros(15),
            np.array([0.3 - delta, 0.2, -0.4]),
            np.array([0.3 - delta, -0.2, -0.4]),
            np.array([-0.3 - delta, 0.2, -0.4]),
            np.array([-0.3 - delta, -0.2, -0.4]),
            np.array([0.5]),
            np.ones(4),
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
        jac_array_x.append(-1 * (unnormalized_gradient[18] + unnormalized_gradient[21] + \
            unnormalized_gradient[24] + unnormalized_gradient[27]))

    ## Margin Y
    out_array_y = []
    jac_array_y = []
    for delta in delta_pos_range_vec:
        network_input = np.concatenate([
            np.array([0.0, 0.0, 1.0]),
            np.zeros(15),
            np.array([0.3 , 0.2 - delta, -0.4]),
            np.array([0.3 , -0.2 - delta, -0.4]),
            np.array([-0.3, 0.2 - delta, -0.4]),
            np.array([-0.3, -0.2 - delta, -0.4]),
            np.array([0.5]),
            np.ones(4),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        network_input = dataset_handler.scale_input(network_input)
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        out_array_y.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))
        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        jac_array_y.append(-1 * (unnormalized_gradient[19] + unnormalized_gradient[22] + \
            unnormalized_gradient[25] + unnormalized_gradient[28]))

    ## Margin LH Swing X
    out_array_LH_x = []
    jac_array_LH_x = []
    for delta in delta_pos_range_vec:
        network_input = np.concatenate([
            np.array([0.0, 0.0, 1.0]),
            np.zeros(15),
            np.array([0.3 - delta, 0.2, -0.4]),
            np.array([0.3 - delta, -0.2, -0.4]),
            np.array([-0.3 - delta, 0.2, -0.4]),
            np.array([-0.3 - delta, -0.2, -0.4]),
            np.array([0.5]),
            np.ones(2),
            np.zeros(1),
            np.ones(1),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        network_input = dataset_handler.scale_input(network_input)
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        out_array_LH_x.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))
        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        jac_array_LH_x.append(-1 * (unnormalized_gradient[18] + unnormalized_gradient[21] + \
            unnormalized_gradient[24] + unnormalized_gradient[27]))

    ## Margin LH Swing Y
    out_array_LH_y = []
    jac_array_LH_y = []
    for delta in delta_pos_range_vec:
        network_input = np.concatenate([
            np.array([0.0, 0.0, 1.0]),
            np.zeros(15),
            np.array([0.3, 0.2 - delta, -0.4]),
            np.array([0.3, -0.2 - delta, -0.4]),
            np.array([-0.3, 0.2 - delta, -0.4]),
            np.array([-0.3, -0.2 - delta, -0.4]),
            np.array([0.5]),
            np.ones(2),
            np.zeros(1),
            np.ones(1),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        network_input = dataset_handler.scale_input(network_input)
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        out_array_LH_y.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))
        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        jac_array_LH_y.append(-1 * (unnormalized_gradient[19] + unnormalized_gradient[22] + \
            unnormalized_gradient[25] + unnormalized_gradient[28]))

    ## Margin Acc X
    out_array_acc_x = []
    jac_array_acc_x = []
    for delta in delta_acc_range_vec:
        network_input = np.concatenate([
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            np.array([0. + delta, 0., 0.]),
            np.zeros(9),
            np.array([0.3, 0.2, -0.4]),
            np.array([0.3, -0.2, -0.4]),
            np.array([-0.3, 0.2, -0.4]),
            np.array([-0.3, -0.2, -0.4]),
            np.array([0.5]),
            np.zeros(1),
            np.ones(2),
            np.zeros(1),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        network_input = dataset_handler.scale_input(network_input)
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        out_array_acc_x.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))
        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        jac_array_acc_x.append(unnormalized_gradient[6])

    ## Margin Acc Y
    out_array_acc_y = []
    jac_array_acc_y = []
    for delta in delta_acc_range_vec:
        network_input = np.concatenate([
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            np.array([0., 0. + delta, 0.]),
            np.zeros(9),
            np.array([0.3, 0.2, -0.4]),
            np.array([0.3, -0.2, -0.4]),
            np.array([-0.3, 0.2, -0.4]),
            np.array([-0.3, -0.2, -0.4]),
            np.array([0.5]),
            np.zeros(1),
            np.ones(2),
            np.zeros(1),
            np.array([0.0, 0.0, 1.0] * 4)
        ])
        network_input = dataset_handler.scale_input(network_input)
        network_input = torch.tensor(network_input, requires_grad=True) # requires_grad to record gradients w.r.t input
        output = network(network_input.float())
        # output = network(torch.from_numpy(network_input).float()).item()
        output = dataset_handler.scale_output(output)
        out_array_acc_y.append(output.detach().numpy()[:][0])
        # Compute gradients of the output w.r.t. the input
        output.backward(torch.ones_like(output))
        unnormalized_gradient = dataset_handler.unscale_gradient(network_input.grad.numpy())
        jac_array_acc_y.append(unnormalized_gradient[7])

    return [[out_array_x, out_array_y], [jac_array_x, jac_array_y],
            [out_array_LH_x, out_array_LH_y], [jac_array_LH_x, jac_array_LH_y],
            [out_array_acc_x, out_array_acc_y], [jac_array_acc_x, jac_array_acc_y]]


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
    comWF_lin_acc = np.array([.0, .0, .0])
    comWF_ang_acc = np.array([.0, .0, .0])

    ''' extForceW is an optional external pure force (no external torque for now) applied on the CoM of the robot.'''
    extForce = np.array([0., .0, 0.0 * 9.81])  # units are N
    extCentroidalTorque = np.array([.0, .0, .0])  # units are Nm
    extCentroidalWrench = np.hstack([extForce, extCentroidalTorque])

    ''' parameters to be tuned'''
    mu = 0.5

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
    LF_foot = np.array([0.3, 0.2, -0.4])
    RF_foot = np.array([0.3, -0.2, -0.4])
    LH_foot = np.array([-0.3, 0.2, -0.4])
    RH_foot = np.array([-0.3, -0.2, -0.4])

    contactsWF = np.vstack((LF_foot, RF_foot, LH_foot, RH_foot))
    params.setContactsPosWF(contactsWF)


    start = time.time()

    # params.useContactTorque = True
    params.useInstantaneousCapturePoint = True
    params.externalCentroidalWrench = extCentroidalWrench
    params.setCoMPosWF(comWF)
    params.comLinVel = [0., 0.0, 0.0]
    params.setCoMLinAcc(comWF_lin_acc)
    params.setTorqueLims(comp_dyn.robotModel.robotModel.joint_torque_limits)
    params.setActiveContacts(stanceFeet)
    params.setConstraintModes(constraint_mode_IP)
    params.setContactNormals(normals)
    params.setFrictionCoefficient(mu)
    params.setTotalMass(comp_dyn.robotModel.robotModel.trunkMass)
    params.externalForceWF = extForceW  # params.externalForceWF is actually used anywhere at the moment

    params_base_roll = deepcopy(params)
    params_base_pitch = deepcopy(params)

    jac = Jacobians(robot_name)
    comp_geom = ComputationalGeometry()


    ''' Margin LH Swing '''
    params_com_x = deepcopy(params)
    params_com_y = deepcopy(params)
    params_com_z = deepcopy(params)
    pos_margin_x, jac_com_pos_x = jac.plotMarginAndJacobianWrtComPosition(params_com_x, delta_pos_range_vec, 0) # dm / dx
    pos_margin_y, jac_com_pos_y = jac.plotMarginAndJacobianWrtComPosition(params_com_y, delta_pos_range_vec, 1) # dm / dy
    pos_margin_z, jac_com_pos_z = jac.plotMarginAndJacobianWrtComPosition(params_com_z, delta_pos_range_vec, 2) # dm / dz

    ''' Margin LH Swing'''
    stanceFeet = [1, 1, 0, 1]
    params.setActiveContacts(stanceFeet)
    params_com_LH_x = deepcopy(params)
    params_com_LH_y = deepcopy(params)
    params_com_LH_z = deepcopy(params)
    pos_margin_LH_x, jac_com_pos_LH_x = jac.plotMarginAndJacobianWrtComPosition(params_com_LH_x, delta_pos_range_vec, 0) # dm / dx
    pos_margin_LH_y, jac_com_pos_LH_y = jac.plotMarginAndJacobianWrtComPosition(params_com_LH_y, delta_pos_range_vec, 1) # dm / dy
    pos_margin_LH_z, jac_com_pos_LH_z = jac.plotMarginAndJacobianWrtComPosition(params_com_LH_z, delta_pos_range_vec, 2) # dm / dz

    ''' Margin Acc'''
    stanceFeet = [0, 1, 1, 0]
    params.setActiveContacts(stanceFeet)
    params_com_acc_x = deepcopy(params)
    params_com_acc_y = deepcopy(params)
    params_com_acc_z = deepcopy(params)
    acc_margin_x, jac_com_lin_acc_x = jac.plotMarginAndJacobianOfMarginWrtComLinAcceleration(params_com_acc_x, delta_acc_range_vec, 0) # dm/d x_dd
    acc_margin_y, jac_com_lin_acc_y = jac.plotMarginAndJacobianOfMarginWrtComLinAcceleration(params_com_acc_y, delta_acc_range_vec, 1) # dm/d y_dd
    acc_margin_z, jac_com_lin_acc_z = jac.plotMarginAndJacobianOfMarginWrtComLinAcceleration(params_com_acc_z, delta_acc_range_vec, 2) # dm/d z_dd
        
    return [[pos_margin_x, pos_margin_y, pos_margin_z], [jac_com_pos_x, jac_com_pos_y, jac_com_pos_z],
            [pos_margin_LH_x, pos_margin_LH_y, pos_margin_LH_z], [jac_com_pos_LH_x, jac_com_pos_LH_y, jac_com_pos_LH_z],
            [acc_margin_x, acc_margin_y, acc_margin_z], [jac_com_lin_acc_x, jac_com_lin_acc_y, jac_com_lin_acc_z]]

#####################################################################################################

def main():

    margin_learnt, jacobian_learnt, margin_LH_learnt, jacobian_LH_learnt, margin_acc_learnt, jacobian_acc_learnt = plot_learnt_margin()
    margin, jacobian, margin_LH, jacobian_LH, margin_acc, jacobian_acc = plot_analytical_margin()
    ### Plotting

    ## X axis
    fig1 = plt.figure(1)
    fig1.suptitle("Stability margin")

    plt.subplot(321)
    plt.plot(delta_pos_range_vec, margin[0], 'g', markersize=15, label='CoM')
    plt.plot(delta_pos_range_vec, margin_learnt[0], 'b', markersize=15, label='CoM')
    plt.grid()
    plt.xlabel("$c_x$ [m]")
    plt.ylabel("m [m]")
    plt.title("CoM X pos margin")

    plt.subplot(322)
    plt.plot(delta_pos_range_vec, margin[1], 'g', markersize=15, label='CoM')
    plt.plot(delta_pos_range_vec, margin_learnt[1], 'b', markersize=15, label='CoM')
    plt.grid()
    plt.xlabel("$c_y$ [m]")
    plt.ylabel("m [m]")
    plt.title("CoM Y pos margin")

    plt.subplot(323)
    plt.plot(delta_pos_range_vec, margin_LH[0], 'g', markersize=15, label='CoM')
    plt.plot(delta_pos_range_vec, margin_LH_learnt[0], 'b', markersize=15, label='CoM')
    plt.grid()
    plt.xlabel("$c_x$ [m]")
    plt.ylabel("m [m]")
    plt.title("CoM X pos LH swing margin")

    plt.subplot(324)
    plt.plot(delta_pos_range_vec, margin_LH[1], 'g', markersize=15, label='CoM')
    plt.plot(delta_pos_range_vec, margin_LH_learnt[1], 'b', markersize=15, label='CoM')
    plt.grid()
    plt.xlabel("$c_y$ [m]")
    plt.ylabel("m [m]")
    plt.title("CoM Y pos LH swing margin")

    plt.subplot(325)
    plt.plot(delta_acc_range_vec, margin_acc[0], 'g', markersize=15, label='CoM')
    plt.plot(delta_acc_range_vec, margin_acc_learnt[0], 'b', markersize=15, label='CoM')
    plt.grid()
    plt.xlabel("$\ddot{c}_x$ [m/s^2]")
    plt.ylabel("m [m]")
    plt.title("CoM X acc margin")

    plt.subplot(326)
    plt.plot(delta_acc_range_vec, margin_acc[1], 'g', markersize=15, label='CoM')
    plt.plot(delta_acc_range_vec, margin_acc_learnt[1], 'b', markersize=15, label='CoM')
    plt.grid()
    plt.xlabel("$\ddot{c}_y$ [m/s^2]")
    plt.ylabel("m [m]")
    plt.title("CoM Y acc margin")

    ## Jacobians
    fig2 = plt.figure(2)
    fig2.suptitle("Gradients")

    plt.subplot(221)
    plt.plot(delta_pos_range_vec, jacobian[0][0], 'g', markersize=15, label='learnt (backprop)')
    plt.plot(delta_pos_range_vec, jacobian_learnt[0], 'b', markersize=15, label='learn (backprop)')
    plt.grid()
    plt.xlabel("$c_x$ [m]")
    plt.ylabel("\delta m/ \delta c_{x}")
    # plt.title("CoM X pos margin")

    plt.subplot(222)
    plt.plot(delta_pos_range_vec, jacobian[1][1], 'g', markersize=15, label='learnt (backprop)')
    plt.plot(delta_pos_range_vec, jacobian_learnt[1], 'b', markersize=15, label='learnt (backprop)')
    plt.grid()
    plt.xlabel("$c_y$ [m]")
    plt.ylabel("\delta m/ \delta c_{y}")
    # plt.title("CoM Y pos margin")

    plt.subplot(223)
    plt.plot(delta_acc_range_vec, jacobian_acc[0][0], 'g', markersize=15, label='learnt (backprop)')
    plt.plot(delta_acc_range_vec, jacobian_acc_learnt[0], 'b', markersize=15, label='learn (backprop)')
    plt.grid()
    plt.xlabel("$\ddot{c}_x$ [m/s^2]")
    plt.ylabel("\delta m/ \delta c_{x}")
    # plt.title("CoM X pos margin")

    plt.subplot(224)
    plt.plot(delta_acc_range_vec, jacobian_acc[1][1], 'g', markersize=15, label='learnt (backprop)')
    plt.plot(delta_acc_range_vec, jacobian_acc_learnt[1], 'b', markersize=15, label='learnt (backprop)')
    plt.grid()
    plt.xlabel("$\ddot{c}_y$ [m/s^2]")
    plt.ylabel("\delta m/ \delta c_{y}")
    # plt.title("CoM Y pos margin")

    plt.show()


if __name__ == '__main__':
    main()