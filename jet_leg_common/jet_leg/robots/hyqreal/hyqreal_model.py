import numpy as np

class HyqrealModel:
    def __init__(self):

        self.trunkMass = 120
        self.trunkInertia = np.array([[4.91182, -0.115619, -0.961905],
                                              [-0.115619,   19.5025, 0.0136923],
                                              [-0.961905, 0.0136923,   21.2841]])

        ''' torque limits for each leg (this code assumes a hyq-like design, i.e. three joints per leg)
        HAA = Hip Abduction Adduction
        HFE = Hip Flextion Extension
        KFE = Knee Flextion Extension
        '''
        LF_tau_lim = [50.0, 100.0, 100.0]  # HAA, HFE, KFE
        RF_tau_lim = [50.0, 100.0, 100.0]  # HAA, HFE, KFE
        LH_tau_lim = [50.0, 100.0, 100.0]  # HAA, HFE, KFE
        RH_tau_lim = [50.0, 100.0, 100.0]  # HAA, HFE, KFE

        self.joint_torque_limits = np.array([LF_tau_lim, RF_tau_lim, LH_tau_lim, RH_tau_lim])
        self.contact_torque_limits = np.array([-1, 1])

        x_nominal_b = 0.44
        y_nominal_b = 0.34
        z_nominal_b = -0.55
        self.nominal_stance_LF = [x_nominal_b, y_nominal_b, z_nominal_b]
        self.nominal_stance_RF = [x_nominal_b, -y_nominal_b, z_nominal_b]
        self.nominal_stance_LH = [-x_nominal_b, y_nominal_b, z_nominal_b]
        self.nominal_stance_RH = [-x_nominal_b, -y_nominal_b, z_nominal_b]
        self.max_dev_from_nominal = [0.15, 0.15, 0.15]