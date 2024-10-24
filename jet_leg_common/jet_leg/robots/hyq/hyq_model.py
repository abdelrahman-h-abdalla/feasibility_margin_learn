import numpy as np

class HyqModel:
    def __init__(self):

        self.trunkMass = 85

        self.trunkInertia = np.array([[0.946438, -0.000938112, 0.00595386],
                                        [-0.000938112, 1.94478, 0.00146328],
                                        [0.00595386, 0.00146328, 2.01835]])

        ''' torque limits for each leg (this code assumes a hyq-like design, i.e. three joints per leg)
        HAA = Hip Abduction Adduction
        HFE = Hip Flextion Extension
        KFE = Knee Flextion Extension
        '''
        LF_tau_lim = [150.0, 150.0, 150.0]  # HAA, HFE, KFE
        RF_tau_lim = [150.0, 150.0, 150.0]  # HAA, HFE, KFE
        LH_tau_lim = [150.0, 150.0, 150.0]  # HAA, HFE, KFE
        RH_tau_lim = [150.0, 150.0, 150.0]  # HAA, HFE, KFE
        self.joint_torque_limits = np.array([LF_tau_lim, RF_tau_lim, LH_tau_lim, RH_tau_lim])
        self.contact_torque_limits = np.array([-5, 5])

        x_nominal_b = 0.36
        y_nominal_b = 0.32
        z_nominal_b = -0.54
        self.nominal_stance_LF = [x_nominal_b, y_nominal_b, z_nominal_b]
        self.nominal_stance_RF = [x_nominal_b, -y_nominal_b, z_nominal_b]
        self.nominal_stance_LH = [-x_nominal_b, y_nominal_b, z_nominal_b]
        self.nominal_stance_RH = [-x_nominal_b, -y_nominal_b, z_nominal_b]
        self.max_dev_from_nominal = [0.15, 0.15, 0.15]