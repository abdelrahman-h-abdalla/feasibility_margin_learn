import os
import numpy as np

from scipy.spatial.transform import Rotation as R

from common.parser import DataParser
from common.paths import ProjectPaths


class TrainingDataset:
    def __init__(self, data_folder='stability_margin'):
        self._data_folder = data_folder
        self._data_parser = None

        self._data_offset = np.concatenate([
            np.array([0, 0, 1]),  # Rotation along gravity axis
            np.zeros(3),  # Linear velocity
            np.zeros(3),  # Linear acceleration
            np.zeros(3),  # Angular acceleration
            np.zeros(3),  # External force
            np.zeros(3),  # External torque
            np.array([0.3, 0.2, -0.45]),  # LF foot
            np.array([0.3, -0.2, -0.45]),  # RF foot
            np.array([-0.3, 0.2, -0.45]),  # LH foot
            np.array([-0.3, -0.2, -0.45]),  # RH foot
            np.array([0.5]),  # Friction
            np.ones(4) * 0.765,  # Feet in contact
            np.array([0, 0, 1] * 4),  # Contact normals
            np.zeros(1)  # Stability margin
        ])

        self._data_multiplier = np.concatenate([
            np.array([1/0.0734, 0.750, 0.003]),  # Rotation along gravity axis
            np.array([0.198, 0.198, 0.053]),  # Linear velocity
            np.array([2.219, 2.203, 0.233]),  # Linear acceleration
            np.array([0.199, 0.199, 0.304]),  # Angular acceleration
            np.array([1.00, 1.00, 1.0]),  # External force
            np.array([1.00, 1.00, 1.0]),  # External torque
            np.ones(1) * 0.152,
            np.ones(1) * 0.085,
            np.ones(1) * 0.077,   # LF foot
            np.ones(1) * 0.152,
            np.ones(1) * 0.085,
            np.ones(1) * 0.077,   # RF foot
            np.ones(1) * 0.152,
            np.ones(1) * 0.085,
            np.ones(1) * 0.077,   # LH foot
            np.ones(1) * 0.152,
            np.ones(1) * 0.085,
            np.ones(1) * 0.077,   # RH foot
            np.ones(1) * 0.112,  # Friction
            np.ones(4) * 0.422,  # Feet in contact
            np.array([0.166, 0.160, 0.0348] * 4),  # Contact normals
            np.ones(1) * 0.111  # Stability margin
        ])

    def get_data_offset(self):
        return self._data_offset

    def get_data_multiplier(self):
        return self._data_multiplier

    def get_data_folder(self):
        return self._data_folder

    def get_training_data_parser(self, process_data=True, max_files=None):
        if self._data_parser is None:
            # Get training data
            data_parser = DataParser()
            paths = ProjectPaths()

            num_files = 0
            num_files_test = 0
            for path, _, files in os.walk(paths.DATA_PATH + '/' + self._data_folder):
                for file_name in files: 
                        num_files_test += 1
            print("Number of files:", num_files_test)
            for path, _, files in os.walk(paths.DATA_PATH + '/' + self._data_folder):
                for file_name in files:
                    if max_files is not None and num_files >= max_files:
                        break

                    if file_name.endswith('.csv'):
                        data_parser.append(filename=os.path.join(path, file_name))

                        num_files += 1
                        print('\rFiles Processed: {}'.format(num_files),)
                if max_files is not None and num_files >= max_files:
                    break
            print

            # Update the data in the data parser object and split it as input and output
            training_data = data_parser.data()

            # Compute rotation matrix (from base to world) from roll/pitch/yaw - Intrinsic rotations.
            rotation_matrices = R.from_euler('XYZ', training_data[:, :3], degrees=False).as_dcm() # 
            rotation_matrices_inv = np.transpose(rotation_matrices, (0, 2, 1))

            # Convert to the base frame
            training_data[:, 18:21] = np.einsum('aik,ak->ai', rotation_matrices_inv,
                                                training_data[:, 18:21])  # LF Foot Position
            training_data[:, 21:24] = np.einsum('aik,ak->ai', rotation_matrices_inv,
                                                training_data[:, 21:24])  # RF Foot Position
            training_data[:, 24:27] = np.einsum('aik,ak->ai', rotation_matrices_inv,
                                                training_data[:, 24:27])  # LH Foot Position
            training_data[:, 27:30] = np.einsum('aik,ak->ai', rotation_matrices_inv,
                                                training_data[:, 27:30])  # RH Foot Position

            # If the foot is not in stance, change its position to nominal
            training_data[np.argwhere(training_data[:, 31] == 0), 18:21] = np.array([0.36, 0.21, -0.47])
            training_data[np.argwhere(training_data[:, 32] == 0), 21:24] = np.array([0.36, -0.21, -0.47])
            training_data[np.argwhere(training_data[:, 33] == 0), 24:27] = np.array([-0.36, 0.21, -0.47])
            training_data[np.argwhere(training_data[:, 34] == 0), 27:30] = np.array([-0.36, -0.21, -0.47])

            # If the foot is not in stance, set its contact normal to vertical
            training_data[np.argwhere(training_data[:, 31] == 0), 35:38] = np.array([0.0, 0.0, 1.0])
            training_data[np.argwhere(training_data[:, 32] == 0), 38:41] = np.array([0.0, 0.0, 1.0])
            training_data[np.argwhere(training_data[:, 33] == 0), 41:44] = np.array([0.0, 0.0, 1.0])
            training_data[np.argwhere(training_data[:, 34] == 0), 44:47] = np.array([0.0, 0.0, 1.0])

            training_data = np.hstack([
                rotation_matrices[:, 2],  # Rotation along gravity axis
                np.einsum('aik,ak->ai', rotation_matrices_inv, training_data[:, 3:6]),  # Linear Velocity
                np.einsum('aik,ak->ai', rotation_matrices_inv, training_data[:, 6:9]),  # Linear Acceleration
                np.einsum('aik,ak->ai', rotation_matrices_inv, training_data[:, 9:12]),  # Angular Acceleration
                np.einsum('aik,ak->ai', rotation_matrices_inv, training_data[:, 12:15]),  # Ext Force in base frame
                np.einsum('aik,ak->ai', rotation_matrices_inv, training_data[:, 15:18]),  # Ext Torque in base frame
                training_data[:, 18:21],  # LF Foot Position: base frame
                training_data[:, 21:24],  # LH Foot Position: base frame
                training_data[:, 24:27],  # RF Foot Position: base frame
                training_data[:, 27:30],  # RH Foot Position: base frame
                training_data[:, 30].reshape(-1, 1),  # Friction
                training_data[:, 31:35],  # Feet in contact
                training_data[:, 35:47],  # Contact normals
                training_data[:, 47].reshape(-1, 1)  # Stability Margin
            ])
            if process_data:
                self.data_offset = data_offset = np.mean(training_data, axis=0)
                self.data_multiplier = data_multiplier = np.std(training_data, axis=0)
                self.data_multiplier[self.data_multiplier == 0] = 1 # Prevent division by 0 for unsampled inputs
                training_data = self.scale_data(training_data)

            data_parser.update_data(training_data)
            data_parser.set_io_split_index(-1)

            # Divide the dataset into training, validation and test
            data_parser.divide_dataset()

            self._data_parser = data_parser
            return data_parser
        else:
            return self._data_parser
        

    def scale_data(self, data):
        return (data - self._data_offset) / self._data_multiplier
    
    def scale_input(self, input):
        return (input - self._data_offset[:-1]) / self._data_multiplier[:-1]
    
    def scale_output(self, output):
        return (output * self._data_offset[:-1]) + self._data_multiplier[:-1]
