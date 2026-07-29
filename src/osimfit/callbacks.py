import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from .model import ModelCache, BodyScaleGroup

###########
# HELPERS #
###########

def _calc_quaternion(state, frame):
    rotation = frame.getRotationInGround(state)
    quaternion = rotation.convertRotationToQuaternion()
    return np.array([quaternion.get(i) for i in range(4)])


def _calc_quaternion_jacobian(eps):
    # Simbody -> /SimTKcommon/Mechanics/include/SimTKcommon/internal/Rotation.h#L712
    e = 0.5 * eps
    return np.array([
        [-e[1], -e[2], -e[3]],
        [ e[0],  e[3], -e[2]],
        [-e[3],  e[0],  e[1]],
        [ e[2], -e[1],  e[0]],
    ])

#########
# TASKS #
#########

class Tasks(ABC):
    """
    A base class for task-specific storage, registration, and error evaluation.
    """
    @abstractmethod
    def initialize_tasks(self, state: osim.State, **kwargs) -> float:
        pass


class MarkerTasks(Tasks):
    """
    Marker-specific task storage, registration, and error evaluation.
    """
    def initialize_tasks(self):
        self.markers = []
        self.mobod_indexes = osim.SimTKArrayInt()
        self.stations = osim.SimTKArrayVec3()
        self.num_tasks: int = 0
        self.positions = []
        self.weights = []

    def add_marker(self, marker_path: str, position: osim.Vec3, weight: float = 1.0):
        """
        Register a marker to track.

        Parameters
        ----------
        marker_path: str
            The OpenSim Model path to the tracking marker.
        position: osim.Vec3
            The reference position data tracked by the model marker.
        weight: float
            (Optional) The cost weight for the position error. Default: 1.0.
        """
        if not self.mc.model.hasComponent(marker_path):
            raise ValueError(f'Model does not have a component at path {marker_path}.')
        if weight < 0:
            raise ValueError(f'Expected weight to be non-negative, but got {weight}.')

        marker = osim.Marker.safeDownCast(self.mc.model.getComponent(marker_path))
        frame = osim.PhysicalFrame.safeDownCast(marker.getParentFrame().findBaseFrame())
        self.mc.model.realizePosition(self.mc.state)
        self.markers.append(marker)
        self.mobod_indexes.push_back(frame.getMobilizedBodyIndex())
        self.stations.push_back(marker.findLocationInFrame(self.mc.state, frame))
        self.num_tasks = self.mobod_indexes.size()
        self.positions.append(position.to_numpy())
        self.weights.append(weight)

    def calc_error(self, state, **kwargs) -> float:
        error = 0.0
        for marker, position, weight in zip(
                self.markers, self.positions, self.weights):
            p_model = marker.getLocationInGround(state).to_numpy()
            error += weight * np.square(np.linalg.norm(p_model - position))
        return error


class FrameTasks(Tasks):
    """
    Frame-specific task storage, registration, and error evaluation.
    """
    def initialize_tasks(self):
        self.frames = []
        self.mobod_indexes = osim.SimTKArrayInt()
        self.stations = osim.SimTKArrayVec3()
        self.num_tasks: int = 0
        self.positions = []
        self.orientations = []
        self.position_weights = []
        self.orientation_weights = []

    def add_frame(self, frame_path: str, position: osim.Vec3,
                  orientation: osim.Quaternion, position_weight: float = 1.0,
                  orientation_weight: float = 1.0):
        """
        Register a frame to track.

        Parameters
        ----------
        frame_path: str
            The OpenSim Model path to the tracking frame.
        position: osim.Vec3
            The reference position data tracked by the model frame.
        orientation: osim.Quaternion
            The reference orientation, expressed as a quaternion, tracked by the model
            frame.
        position_weight: float
            (Optional) The cost weight for the position error. Default: 1.0.
        orientation_weight: float
            (Optional) The cost weight for the orientation error. Default: 1.0.
        """
        if not self.mc.model.hasComponent(frame_path):
            raise ValueError(f'Model does not have a component at path {frame_path}.')
        if position_weight < 0:
            raise ValueError(f'Expected position_weight to be non-negative, but got '
                             f'{position_weight}.')
        if orientation_weight < 0:
            raise ValueError(f'Expected orientation_weight to be non-negative, but got '
                             f'{orientation_weight}.')

        frame = osim.PhysicalFrame.safeDownCast(self.mc.model.getComponent(frame_path))
        self.frames.append(frame)
        self.mobod_indexes.push_back(frame.getMobilizedBodyIndex())
        self.stations.push_back(osim.Vec3(frame.findTransformInBaseFrame().p()))
        self.num_tasks = self.mobod_indexes.size()
        self.positions.append(position.to_numpy())
        self.orientations.append(np.array([orientation.get(i) for i in range(4)]))
        self.position_weights.append(position_weight)
        self.orientation_weights.append(orientation_weight)

    def calc_error(self, state, **kwargs) -> float:
        error = 0.0
        for i, frame in enumerate(self.frames):
            p_model = frame.getPositionInGround(state).to_numpy()
            position_error = self.position_weights[i] * np.square(
                np.linalg.norm(p_model - self.positions[i]))

            eps = _calc_quaternion(state, frame)
            orientation_error = self.orientation_weights[i] * (
                1.0 - np.square(np.dot(eps, self.orientations[i])))

            error += position_error + orientation_error
        return error


#########
# COSTS #
#########

class TrackingCost(ABC):
    """
    A base class for tracking cost functions that compute a scalar error and its
    Jacobian with respect to the model's generalized coordinates, body scales,
    and other optimization variables. To implement a new tracking cost, extend this
    class and implement the abstract methods (calc_error, calc_jacobian) to compute the
    error and its Jacobian.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def calc_error(self, state: osim.State, **kwargs) -> float:
        pass

    @abstractmethod
    def calc_jacobian(self, state: osim.State, **kwargs) -> list[np.ndarray]:
        pass


class BilevelCost(TrackingCost):
    """
    An intermediate base class that provides functionality common to bilevel
    optimization costs.
    """
    def __init__(self):
        super().__init__()

    def apply_scales(self, body_scales: np.ndarray, state: osim.State) -> None:
        """
        Apply body-scale overrides to `state`. Invalidates Stage::Instance and higher.

        Parameters
        ----------
        body_scales: np.ndarray, shape (3 * len(body_scale_groups),)
            XYZ body-scale variables, one 3-vector per BodyScaleGroup.
        state: osim.State
            The State to update.
        """
        # Apply inboard and outboard frame positions to the model based on the current
        # set of body scales.
        self.mc.set_scaled_mobilizer_frame_positions(state, self.body_scale_groups,
                                                        body_scales)


class FrameTrackingCost(FrameTasks, TrackingCost):
    """
    A tracking cost that computes the aggregate error between model frames' positions
    and orientations and corresponding reference data as a function of the model's
    generalized coordinates. Individual frames are registered via add_frame().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        if self.num_tasks == 0:
            return [np.zeros((1, len(self.mc.q_indexes)))]

        # Loop over all frames and compute the "spatial error" (i.e., the combined
        # position and orientation error) for each.
        spatialError = osim.VectorOfSpatialVec(self.num_tasks, osim.SpatialVec(0))
        for i, frame in enumerate(self.frames):
            wp = self.position_weights[i]
            wo = self.orientation_weights[i]

            # Position error.
            p_model = frame.getPositionInGround(state)
            p_error = osim.Vec3(
                2.0 * wp * (p_model[0] - self.positions[i][0]),
                2.0 * wp * (p_model[1] - self.positions[i][1]),
                2.0 * wp * (p_model[2] - self.positions[i][2]))

            # Orientation error.
            eps = _calc_quaternion(state, frame)
            jac_eps = _calc_quaternion_jacobian(eps)
            omega = jac_eps.T @ self.orientations[i]
            scale = wo * -2.0 * np.dot(eps, self.orientations[i])
            w_error = osim.Vec3(scale * omega[0], scale * omega[1], scale * omega[2])

            # Combine the position and orientation into a SpatialVec to pass to the
            # frame Jacobian operator below.
            spatialError.set(i, osim.SpatialVec(w_error, p_error))

        # Calculate the frame (position and orientation) error Jacobian.
        vec = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, vec)
        J = vec.to_numpy()

        return [np.expand_dims(J[self.mc.q_indexes], axis=0)]


class MarkerTrackingCost(MarkerTasks, TrackingCost):
    """
    A tracking cost that computes the aggregate error between model markers' positions
    and corresponding reference positions as a function of the model's generalized
    coordinates. Individual markers are registered via add_marker().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        if self.num_tasks == 0:
            return [np.zeros((1, len(self.mc.q_indexes)))]

        # Inialize the array used to calculate the position error Jacobian via the
        # grouped Simbody operator.
        f_GP = osim.VectorVec3(self.num_tasks, osim.Vec3(0))
        for i, (marker, position, weight) in enumerate(
                zip(self.markers, self.positions, self.weights)):
            p_model = marker.getLocationInGround(state)
            f_GP.set(i, osim.Vec3(
                2.0 * weight * (p_model[0] - position[0]),
                2.0 * weight * (p_model[1] - position[1]),
                2.0 * weight * (p_model[2] - position[2])))

        # Calculate the position error Jacobian.
        vec = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, f_GP, vec)

        return [np.expand_dims(vec.to_numpy()[self.mc.q_indexes], axis=0)]


class MarkerBilevelCost(MarkerTasks, BilevelCost):
    """
    A tracking cost that computes the aggregate error between model markers' scaled
    positions and corresponding reference positions as a function of the model's
    generalized coordinates and body scales. Individual markers are registered via
    add_marker().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    body_scale_groups: list[BodyScaleGroup]
        Groups of bodies each sharing one set of XYZ body scales. Each entry
        contains a list of mobilized body indexes defining which bodies are scaled and
        how Jacobian columns are aggregated.
    """
    def __init__(self, mc: ModelCache, body_scale_groups: list[BodyScaleGroup]):
        self.mc = mc
        self.initialize_tasks()
        self.body_scale_groups = body_scale_groups

        # Cache references to joints associated with each inboard and outboard frame in
        # each scale group.
        for group in self.body_scale_groups:
            group.outboard_joints = [
                self.mc.get_joint_for_mobilized_body_index(int(k))
                for k in group.mobod_indexes]
            group.inboard_joints = [
                self.mc.get_joint_for_mobilized_body_index(c)
                for k in group.mobod_indexes
                for c in self.mc.children_of[int(k)]]

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        Jq = np.zeros((1, len(self.mc.q_indexes)))
        Js = np.zeros((1, 3 * len(self.body_scale_groups)))
        if self.num_tasks == 0:
            return [Jq, Js]

        # Calculate the per-marker error gradient in Ground. This is a force-like term
        # will be multiplied with (the transpose of) each position Jacobian below.
        dp_GS = osim.VectorVec3(self.num_tasks, osim.Vec3(0))
        for i, (marker, position, weight) in enumerate(
                zip(self.markers, self.positions, self.weights)):
            p_GS = marker.getLocationInGround(state)
            dp_GS.set(i, osim.Vec3(2.0 * weight * (p_GS[0] - position[0]),
                                   2.0 * weight * (p_GS[1] - position[1]),
                                   2.0 * weight * (p_GS[2] - position[2])))

        # Calculate the Jacobian of the position error with respect to the coordinates.
        vec = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, dp_GS, vec)
        Jq[0, :] = vec.to_numpy()[self.mc.q_indexes]

        # Scatter per-station gradients for each task into a vector respresenting the
        # error gradient with respect to body origins, which we need for the Jacobian
        # operations below. Since the body scales only apply a translational shift and
        # no rotation, `dp_GS_i / dp_GB[k_i] = I`, and we can compute the vector via:
        #
        #     dp_GB.get(k) += dp_GS.get(i)   # for each marker i on body k
        #
        dp_GB = osim.VectorVec3(self.mc.num_mobod, osim.Vec3(0))
        for i in range(self.num_tasks):
            k = int(self.mobod_indexes.getElt(i))
            cur = dp_GB.get(k).to_numpy() + dp_GS.get(i).to_numpy()
            dp_GB.set(k, osim.Vec3(float(cur[0]), float(cur[1]), float(cur[2])))

        # Calculate the position-error Jacobian with respect to body scales.
        Js = self.mc.calc_position_jacobian_wrt_body_scales(state, dp_GB,
                                                              self.body_scale_groups)

        return [Jq, Js]


class FrameBilevelCost(FrameTasks, BilevelCost):
    """
    A tracking cost that computes the aggregate error between model frames' scaled
    positions and corresponding reference positions as a function of the model's
    generalized coordinates and body scales. Individual frames are registered via
    add_frame().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    body_scale_groups: list[BodyScaleGroup]
        Groups of bodies each sharing one set of XYZ body scales. Each entry
        contains a list of mobilized body indexes defining which bodies are scaled and
        how Jacobian columns are aggregated.
    """
    def __init__(self, mc: ModelCache, body_scale_groups: list[BodyScaleGroup]):
        self.mc = mc
        self.initialize_tasks()
        self.body_scale_groups = body_scale_groups

        # Cache references to joints associated with each inboard and outboard frame in
        # each scale group.
        for group in self.body_scale_groups:
            group.outboard_joints = [
                self.mc.get_joint_for_mobilized_body_index(int(k))
                for k in group.mobod_indexes]
            group.inboard_joints = [
                self.mc.get_joint_for_mobilized_body_index(c)
                for k in group.mobod_indexes
                for c in self.mc.children_of[int(k)]]

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        Jq = np.zeros((1, len(self.mc.q_indexes)))
        Js = np.zeros((1, 3 * len(self.body_scale_groups)))
        if self.num_tasks == 0:
            return [Jq, Js]

        # Loop over all frames and compute the "spatial error" (i.e., the combined
        # position and orientation error) for each.
        spatialError = osim.VectorOfSpatialVec(self.num_tasks, osim.SpatialVec(0))
        # Store the position-error gradient along the way. We need it for the body scale
        # Jacobian calculations.
        dp_GF = osim.VectorVec3(self.num_tasks, osim.Vec3(0))
        for i, frame in enumerate(self.frames):
            wp = self.position_weights[i]
            wo = self.orientation_weights[i]
            position = self.positions[i]

            p_GF = frame.getPositionInGround(state)
            dp_GF.set(i, osim.Vec3(2.0 * wp * (p_GF[0] - position[0]),
                                   2.0 * wp * (p_GF[1] - position[1]),
                                   2.0 * wp * (p_GF[2] - position[2])))

            # Calculate the per-frame orientation error in Ground.
            eps = _calc_quaternion(state, frame)
            jac_eps = _calc_quaternion_jacobian(eps)
            omega = jac_eps.T @ self.orientations[i]
            scale = wo * -2.0 * np.dot(eps, self.orientations[i])
            dw_GF = osim.Vec3(scale * omega[0], scale * omega[1], scale * omega[2])

            # Combine the position and orientation into a SpatialVec to pass to the
            # frame Jacobian operator below.
            spatialError.set(i, osim.SpatialVec(dw_GF, dp_GF.get(i)))

        # Calculate the frame (position and orientation) error Jacobian.
        vec = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, vec)
        Jq[0, :] = vec.to_numpy()[self.mc.q_indexes]

        # Scatter per-station gradients for each task into a vector respresenting the
        # error gradient with respect to body origins, which we need for the Jacobian
        # operations below. Since the body scales only apply a translational shift and
        # no rotation, `dp_GF_i / dp_GB[k_i] = I`, and we can compute the vector via:
        #
        #     dp_GB.get(k) += dp_GF.get(i)   # for each frame i on body k
        #
        dp_GB = osim.VectorVec3(self.mc.num_mobod, osim.Vec3(0))
        for i in range(self.num_tasks):
            k = int(self.mobod_indexes.getElt(i))
            cur = dp_GB.get(k).to_numpy() + dp_GF.get(i).to_numpy()
            dp_GB.set(k, osim.Vec3(float(cur[0]), float(cur[1]), float(cur[2])))

        # Calculate the position-error Jacobian with respect to body scales.
        Js = self.mc.calc_position_jacobian_wrt_body_scales(state, dp_GB,
                                                            self.body_scale_groups)

        return [Jq, Js]


class Function(ca.Callback, ABC):
    """
    A base class for CasADi callback functions that evaluate the function and its
    Jacobian using OpenSim. To implement a new callback, extend this class and implement
    the abstract methods to define the number of inputs and outputs and provide the
    function evaluation and its Jacobian.

    Parameters
    ----------
    name: str
        The name of the callback function.
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function
        and its Jacobian and caching model information.
    opts: dict
        A dictionary of options to pass to the CasADi callback constructor.
    """
    def __init__(self, name: str, mc: ModelCache, opts: dict = {}):
        ca.Callback.__init__(self)
        self.mc = mc
        self.state = self.mc.state
        self.enable_fd = opts.get("enable_fd", False)
        self.construct(name, opts)

    def get_n_in(self): return self._get_num_inputs()
    def get_n_out(self): return self._get_num_outputs()

    def get_input_size(self, i):
        return self._get_input_size(i)

    def get_output_size(self, i):
        return self._get_output_size(i)

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.get_input_size(i), 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.get_output_size(i), 1)

    def eval(self, arg):
        return self._eval(arg)

    def has_jacobian(self): return not self.enable_fd

    def get_jacobian(self, name, inames, onames, opts):
        class JacobianFunction(ca.Callback):
            def __init__(self, callback, opts={}):
                ca.Callback.__init__(self)
                self.callback = callback
                self.construct(name, opts)

            def get_n_in(self):
                return self.callback.get_n_in() + self.callback.get_n_out()
            def get_n_out(self):
                return self.callback.get_n_in()

            def get_sparsity_in(self,i):
                if i < self.callback.get_n_in():
                    return ca.Sparsity.dense(self.callback.get_input_size(i), 1)
                elif i < self.callback.get_n_in() + self.callback.get_n_out():
                    iout = i - self.callback.get_n_in()
                    return ca.Sparsity.dense(self.callback.get_output_size(iout), 1)
                else:
                    return ca.Sparsity.dense(0, 0)

            def get_sparsity_out(self,i):
                iin = i % self.callback.get_n_in()
                iout = i // self.callback.get_n_in()
                return ca.Sparsity.dense(self.callback.get_output_size(iout),
                                         self.callback.get_input_size(iin))

            def eval(self, arg):
                return self.callback._jac_eval(arg)

        self.jacobian_callback = JacobianFunction(self)
        return self.jacobian_callback

    @abstractmethod
    def _get_num_inputs(self):
        pass

    @abstractmethod
    def _get_num_outputs(self):
        pass

    @abstractmethod
    def _get_input_size(self, i):
        pass

    @abstractmethod
    def _get_output_size(self, i):
        pass

    @abstractmethod
    def _eval(self, arg):
        pass

    @abstractmethod
    def _jac_eval(self, arg):
        pass


##################
# COST FUNCTIONS #
##################

class TrackingCostFunction(Function):
    """
    A CasADi callback that evaluates the sum of tracking costs over a set of model
    frames and markers with respect to the model's generalized coordinates.

    Parameters
    ----------
    name: str
        The name of the callback function.
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function and
        its Jacobian and caching model information.
    opts: dict
        A dictionary of options to pass to the CasADi callback constructor.
    """
    def __init__(self, name: str, mc: ModelCache, opts={}):
        Function.__init__(self, name, mc, opts=opts)
        self.marker_cost = MarkerTrackingCost(mc)
        self.frame_cost = FrameTrackingCost(mc)

    def apply_state(self, arg):
        """
        Apply the input coordinates to the model state and realize the system to the
        position stage.
        """
        q = np.zeros(self.state.getNQ())
        q[self.mc.q_indexes] = np.squeeze(arg[0].full())
        self.state.setQ(osim.Vector.createFromMat(q))
        self.mc.model.realizePosition(self.state)

    def add_marker_tracking_cost(self, marker_path: str, position: osim.Vec3,
                                 weight: float = 1.0):
        self.marker_cost.add_marker(marker_path, position, weight=weight)

    def add_frame_tracking_cost(self, frame_path: str,
                                position: osim.Vec3,
                                orientation: osim.Quaternion,
                                position_weight: float = 1.0,
                                orientation_weight: float = 1.0):
        self.frame_cost.add_frame(frame_path, position, orientation,
                                  position_weight=position_weight,
                                  orientation_weight=orientation_weight)

    def _get_num_inputs(self):
        return 1

    def _get_num_outputs(self):
        return 1

    def _get_input_size(self, i):
        if i == 0:
            return len(self.mc.q_indexes)
        else:
            raise IndexError(f'Invalid input index {i} for TrackingCostFunction.')

    def _get_output_size(self, i):
        if i == 0:
            return 1
        else:
            raise IndexError(f'Invalid output index {i} for TrackingCostFunction.')

    def _eval(self, arg):
        self.apply_state(arg)
        error = (self.marker_cost.calc_error(self.state) +
                 self.frame_cost.calc_error(self.state))
        return [error]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        J = (self.marker_cost.calc_jacobian(self.state)[0] +
             self.frame_cost.calc_jacobian(self.state)[0])
        return [J]


class BilevelCostFunction(Function):
    """
    A CasADi callback that evaluates the sum of tracking costs over a set of model
    markers with respect to the model's generalized coordinates and a set of body
    scales.

    Parameters
    ----------
    name: str
        The name of the callback function.
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function and
        its Jacobian and caching model information.
    body_scale_groups: list[BodyScaleGroup]
        Groups of bodies each sharing one set of XYZ body scales. The i-th 3-vector of
        body scales is broadcast to every body in `body_scale_groups[i]`.
    opts: dict
        A dictionary of options to pass to the CasADi callback constructor.
    """
    def __init__(self, name: str, mc: ModelCache,
                 body_scale_groups: list[BodyScaleGroup], opts={}):
        self.body_scale_groups = body_scale_groups
        Function.__init__(self, name, mc, opts=opts)
        self.marker_cost = MarkerBilevelCost(mc, self.body_scale_groups)
        self.frame_cost = FrameBilevelCost(mc, self.body_scale_groups)

    def apply_state(self, arg):
        """
        Apply input coordinates and body-scale variables to the model State, then
        realize to Position.
        """
        body_scales = np.squeeze(arg[1].full())
        body_scales = np.atleast_1d(body_scales).astype(float)
        self.marker_cost.apply_scales(body_scales, self.state)
        self.frame_cost.apply_scales(body_scales, self.state)

        q = np.zeros(self.state.getNQ())
        q[self.mc.q_indexes] = np.squeeze(arg[0].full())
        self.state.setQ(osim.Vector.createFromMat(q))
        self.mc.model.realizePosition(self.state)

    def add_marker_bilevel_cost(self, marker_path: str, position: osim.Vec3,
                                weight: float = 1.0):
        self.marker_cost.add_marker(marker_path, position, weight=weight)

    def add_frame_bilevel_cost(self, frame_path: str, position: osim.Vec3,
                               orientation: osim.Quaternion,
                               position_weight: float = 1.0,
                               orientation_weight: float = 1.0):
        self.frame_cost.add_frame(frame_path, position, orientation, position_weight,
                                  orientation_weight)

    def _get_num_inputs(self):
        return 2

    def _get_num_outputs(self):
        return 1

    def _get_input_size(self, i):
        if i == 0:
            return len(self.mc.q_indexes)
        elif i == 1:
            return 3 * len(self.body_scale_groups)
        else:
            raise IndexError(f'Invalid input index {i} for BilevelCostFunction.')

    def _get_output_size(self, i):
        if i == 0:
            return 1
        else:
            raise IndexError(f'Invalid output index {i} for BilevelCostFunction.')

    def _eval(self, arg):
        self.apply_state(arg)
        error = 0
        error += self.marker_cost.calc_error(self.state)
        error += self.frame_cost.calc_error(self.state)
        return [error]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        marker_jac = self.marker_cost.calc_jacobian(self.state)
        frame_jac = self.frame_cost.calc_jacobian(self.state)
        return [Jm + Jf for Jm, Jf in zip(marker_jac, frame_jac)]
