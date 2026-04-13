import os
import time
import osimfit
import opensim as osim
from copy import deepcopy


model = osim.Model('unscaled_generic.osim')
c3d_source = osimfit.C3DSource('pose_0.c3d')
scaler = osimfit.ModelScaler(model, c3d_source)


scaler.register_frame('l_shank', '/jointset/walker_knee_l/tibia_l_offset/l_shank')
scaler.register_frame('l_foot', '/jointset/ankle_l/talus_l_offset/l_foot')
scaler.register_frame('l_toes', '/jointset/mtp_l/toes_l_offset/l_toes')
scaler.register_frame('r_thigh', '/jointset/hip_r/femur_r_offset/r_thigh')
scaler.register_frame('r_shank', '/jointset/walker_knee_r/tibia_r_offset/r_shank')
scaler.register_frame('r_foot', '/jointset/ankle_r/talus_r_offset/r_foot')
scaler.register_frame('r_toes', '/jointset/mtp_r/toes_r_offset/r_toes')
scaler.register_frame('l_uarm', '/jointset/acromial_l/humerus_l_offset/l_uarm')
scaler.register_frame('l_larm', '/jointset/elbow_l/ulna_l_offset/l_larm')
scaler.register_frame('l_thigh', '/jointset/hip_l/femur_l_offset/l_thigh')
scaler.register_frame('l_hand', '/jointset/radius_hand_l/hand_l_offset/l_hand')
scaler.register_frame('r_uarm', '/jointset/acromial_r/humerus_r_offset/r_uarm')
scaler.register_frame('r_larm', '/jointset/elbow_r/ulna_r_offset/r_larm')
scaler.register_frame('r_hand', '/jointset/radius_hand_r/hand_r_offset/r_hand')
scaler.register_frame('pelvis', '/bodyset/pelvis/pelvis')
scaler.register_frame('torso',  '/bodyset/torso/torso')


scaler.add_frame_measurement('pelvis', 'pelvis', 'torso', osimfit.Axis.YAxis)
scaler.add_frame_measurement('pelvis', 'l_thigh', 'r_thigh', osimfit.Axis.ZAxis)
scaler.add_frame_measurement('torso', 'torso', 'pelvis', osimfit.Axis.YAxis)
scaler.add_frame_measurement('torso', 'l_uarm', 'r_uarm', osimfit.Axis.ZAxis)
scaler.add_frame_measurement('humerus_r', 'r_uarm', 'r_larm', osimfit.Axis.YAxis)
scaler.add_frame_measurement('humerus_l', 'l_uarm', 'l_larm', osimfit.Axis.YAxis)
scaler.add_frame_measurement('radius_r', 'r_larm', 'r_hand', osimfit.Axis.YAxis)
scaler.add_frame_measurement('radius_l', 'l_larm', 'l_hand', osimfit.Axis.YAxis)
scaler.add_frame_measurement('femur_r', 'r_thigh', 'r_shank', osimfit.Axis.YAxis)
scaler.add_frame_measurement('femur_l', 'l_thigh', 'l_shank', osimfit.Axis.YAxis)
scaler.add_frame_measurement('tibia_r', 'r_shank', 'r_foot', osimfit.Axis.YAxis)
scaler.add_frame_measurement('tibia_l', 'l_shank', 'l_foot', osimfit.Axis.YAxis)
scaler.add_frame_measurement('calcn_r', 'r_foot', 'r_toes', osimfit.Axis.XAxis)
scaler.add_frame_measurement('calcn_r', 'r_foot', 'r_toes', osimfit.Axis.YAxis)
scaler.add_frame_measurement('calcn_l', 'l_foot', 'l_toes', osimfit.Axis.XAxis)
scaler.add_frame_measurement('calcn_l', 'l_foot', 'l_toes', osimfit.Axis.YAxis)


scaler.add_symmetry_pair('l_uarm', 'r_uarm')
scaler.add_symmetry_pair('l_larm', 'r_larm')
scaler.add_symmetry_pair('l_thigh', 'r_thigh')
scaler.add_symmetry_pair('l_shank', 'r_shank')
scaler.add_symmetry_pair('l_foot', 'r_foot')
scaler.add_symmetry_pair('l_toes', 'r_toes')


scaled_model  = scaler.scale()
scaled_model.printToXML('jump_1_scaled.osim')



# # Step 4: Adjust anthropometry.
# # ------------------------------
# scaled_model_fpath = os.path.join(trial_path, f'{scaled_model_name}.osim')
# anthropometrics_fpath = os.path.join('anthropometrics', 'ANSUR_II_FEMALE_Public.csv')
# adjusted_model_fpath = os.path.join(trial_path, f'{scaled_model_name}_adjusted.osim')
# adjust_anthropometry(scaled_model_fpath, anthropometrics_fpath, adjusted_model_fpath)

# # Step 5: Inverse kinematics.
# # ---------------------------
# # Using CasADi (https://web.casadi.org/), create a custom inverse kinematics problem
# # that minimizes the error between model and Theia frame positions and orientations.

# # Cost function weights.
# # - position: squared norm of frame position errors (in m^2)
# # - orientation: quaternion distance of frame orientation errors (between [0, 1])
# # - smoothness: sum of squared differences in generalized coordinates between current
# #               and previous time step (in rad^2 or m^2)
# weights = {'position': 2.0,
#            'orientation': 5.0,
#            'smoothness': 0.5}

# # Convergence tolerance: controls various IPOPT tolerances (e.g., primal and dual
# # feasibility, acceptable tol, etc.).
# convergence_tolerance = 1e-4

# # Whether or not to compute the tracking cost derivative using finite differences. By
# # default, we set this to False to use an analytical derivatives computed from Simbody,
# # which is roughly 10X faster.
# finite_differences = False

# # Run inverse kinematics.
# start_time = time.time()
# run_inverse_kinematics(adjusted_model_fpath, trial_path, c3d_filename, offset_frame_map,
#                        weights, convergence_tolerance,
#                        finite_differences=finite_differences)
# end_time = time.time()
# print(f"Inverse kinematics took {end_time - start_time:.2f} seconds")
