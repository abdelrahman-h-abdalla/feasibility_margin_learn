import numpy as np

class AnymalModel:
    def __init__(self, anymal_type):
        self.anymal_type = anymal_type
        
        if self.anymal_type == 'anymal_boxy':
            self.trunkMass = 35
            self.trunkInertia = np.array([[0.946438, -0.000938112, 0.00595386],
                                          [-0.000938112, 1.94478, 0.00146328],
                                          [0.00595386, 0.00146328, 2.01835]])

            ''' torque limits for each leg (this code assumes three joints per leg)
            HAA = Hip Abduction Adduction
            HFE = Hip Flextion Extension
            KFE = Knee Flextion Extension
            '''
            LF_tau_lim = [40.0, 40.0, 40.0]  # HAA, HFE, KFE
            RF_tau_lim = [40.0, 40.0, 40.0]  # HAA, HFE, KFE
            LH_tau_lim = [40.0, 40.0, 40.0]  # HAA, HFE, KFE
            RH_tau_lim = [40.0, 40.0, 40.0]  # HAA, HFE, KFE
            self.joint_torque_limits = np.array([LF_tau_lim, RF_tau_lim, LH_tau_lim, RH_tau_lim])
            self.contact_torque_limits = np.array([-1.5, 1.5])

            x_nominal_b = 0.36
            y_nominal_b = 0.2
            z_nominal_b = -0.4
            self.nominal_stance_LF = [x_nominal_b, y_nominal_b, z_nominal_b]
            self.nominal_stance_RF = [x_nominal_b, -y_nominal_b, z_nominal_b]
            self.nominal_stance_LH = [-x_nominal_b, y_nominal_b, z_nominal_b]
            self.nominal_stance_RH = [-x_nominal_b, -y_nominal_b, z_nominal_b]
            self.max_dev_from_nominal = [0.1, 0.1, 0.1]

        elif self.anymal_type == 'anymal_coyote':
            self.trunkMass = 52
            self.trunkInertia = np.array([[0.946438, -0.000938112, 0.00595386],
                                          [-0.000938112, 1.94478, 0.00146328],
                                          [0.00595386, 0.00146328, 2.01835]])
            
            ''' torque limits for each leg (this code assumes three joints per leg)
            HAA = Hip Abduction Adduction
            HFE = Hip Flextion Extension
            KFE = Knee Flextion Extension
            '''
            LF_tau_lim = [80.0, 80.0, 80.0]  # HAA, HFE, KFE
            RF_tau_lim = [80.0, 80.0, 80.0]  # HAA, HFE, KFE
            LH_tau_lim = [80.0, 80.0, 80.0]  # HAA, HFE, KFE
            RH_tau_lim = [80.0, 80.0, 80.0]  # HAA, HFE, KFE
            self.joint_torque_limits = np.array([LF_tau_lim, RF_tau_lim, LH_tau_lim, RH_tau_lim])
            self.contact_torque_limits = np.array([-1.5, 1.5])

            x_nominal_b = 0.29
            y_nominal_b = 0.19
            z_nominal_b = -0.46
            self.nominal_stance_LF = [x_nominal_b, y_nominal_b, z_nominal_b]
            self.nominal_stance_RF = [x_nominal_b, -y_nominal_b, z_nominal_b]
            self.nominal_stance_LH = [-x_nominal_b, y_nominal_b, z_nominal_b]
            self.nominal_stance_RH = [-x_nominal_b, -y_nominal_b, z_nominal_b]
            self.max_dev_from_nominal = [0.2, 0.1, 0.1]