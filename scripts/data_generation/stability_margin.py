import os
import time
import string
import psutil
import multiprocessing

from common.parameters import *
from common.jet_leg_interface import compute_stability
from common.paths import ProjectPaths

from jet_leg_common.jet_leg.dynamics.computational_dynamics import ComputationalDynamics
from jet_leg_common.jet_leg.computational_geometry.computational_geometry import ComputationalGeometry
from jet_leg_common.jet_leg.computational_geometry.iterative_projection_parameters import IterativeProjectionParameters

CONSTRAINTS = 'FRICTION_AND_ACTUATION'
MAX_ITERATIONS = 500000
COMPUTE_JACOBIAN = False
STORE_BINARY_MATRIX = False


paths = ProjectPaths()
save_path = paths.DATA_PATH + '/stability_margin/' + paths.INIT_DATETIME_STR + '/' + ''.join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
file_name = 'data_'
robot_name = 'anymal_coyote'

file_suffix = multiprocessing.Value('i', 0)
exceptions = multiprocessing.Value('i', 0)
successes = multiprocessing.Value('i', 0)

start_time = None

def computation(i, q):

    global paths, save_path, file_name, robot_name
    global file_suffix, exceptions, successes

    start_time = time.time()
    math_tools = Math()
    comp_dyn = ComputationalDynamics(robot_name)
    params = IterativeProjectionParameters(robot_name=robot_name)
    comp_geom = ComputationalGeometry()
    seed_random()
    stance_feet_list = stance_feet(high=2)  # Get random stance configuration
    com_world = com_positions(robot_name)  # Center of Mass in world frame
    com_lin_vel = linear_velocity()  # Random Linear Velocity
    com_lin_acc = linear_acceleration()  # Random COM linear acceleration
    com_ang_acc = angular_acceleration()  # Random COM angular acceleration
    friction = friction_coeff()  # Gazebo uses 0.8
    feet_pos = feet_positions(robot_name)  # Random positions of foot contacts
    feet_contact_normals = contact_normals(math_tools=math_tools)  # List of Normal Rotations for each contact
    ext_force_world = base_external_force()  # External force applied to base in world frame
    ext_torque_world = base_external_torque()
    com_euler = base_euler()
    parameters = []
    results = []
    success = False
    try:
        stability_margin = compute_stability(
            comp_dyn=comp_dyn,
            params=params,
            comp_geom=comp_geom,
            constraint_mode=CONSTRAINTS,
            com=com_world,
            com_euler=com_euler,
            com_lin_vel=com_lin_vel,
            com_lin_acc=com_lin_acc,
            com_ang_acc=com_ang_acc,
            ext_force=ext_force_world,
            ext_torque=ext_torque_world,
            feet_position=feet_pos,
            mu=friction,
            stance_feet=stance_feet_list,
            contact_normals=feet_contact_normals
        )
        print ('Stability Margin:', stability_margin)
        parameters = np.concatenate([
            np.array(com_euler).flatten(),
            np.array(com_lin_vel).flatten(),
            np.array(com_lin_acc).flatten(),
            np.array(com_ang_acc).flatten(),
            np.array(ext_force_world).flatten(),
            np.array(ext_torque_world).flatten(),
            np.array(feet_pos).flatten(),
            np.array([friction]).flatten(),
            np.array(stance_feet_list).flatten(),
            np.array(feet_contact_normals).flatten(),
        ])
        parameters = np.array(parameters, dtype=np.float)
        results = np.concatenate([
            np.array([stability_margin]).flatten()
        ])
        results = np.array(results, dtype=np.float)

        successes.value += 1
        success = True
        # Create a new file after every 50k samples
        if successes.value % 50000 == 0:
            file_suffix.value += 1
    except Exception as e:
        exceptions.value += 1
        print ('Exception Occurred ', e)

    print ('\n\n----\nIteration', i + 1, 'of', MAX_ITERATIONS, '| Successes:', successes.value, '| Exceptions:', exceptions.value, \
    ' | Elapsed Time:', time.time() - start_time, 'seconds')
    q.put([np.concatenate([parameters, results]), success])

def listener(q):
    '''listens for messages on the q.
     Save to file. Create it if it doesn't exist. '''
    
    while True:
        results = q.get()
        if results == 'kill':
            break
        elif results[1]:
            f = open(save_path + '/' + file_name + str(file_suffix.value).zfill(4) + '.csv', 'ab')
            np.savetxt(f, results[0].flatten(), delimiter=',', fmt='%1.6f', newline=',')
            f.seek(-1, os.SEEK_END)
            f.truncate()
            f.write(b"\n")
            f.close()

def main():

    global save_path
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    num_cpu = psutil.cpu_count(True)

    with multiprocessing.Pool(num_cpu) as pool:
        manager = multiprocessing.Manager() # To manage the writing to file
        q = manager.Queue() 
        global start_time
        start_time = time.time()
        watcher = pool.apply_async(listener, (q,))
        jobs = [pool.apply_async(computation, args=(i, q,))
                     for i in range(MAX_ITERATIONS)]
        [j.get() for j in jobs]
        q.put('kill')
    # pool.close()
    # pool.join()
    
    print ('\n\nTotal Execution Time:', time.time() - start_time, 'seconds\n')


if __name__ == '__main__':
    main()
