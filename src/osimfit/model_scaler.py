from abc import ABC, abstractmethod
import os
import numpy as np
import opensim as osim
from enum import Enum
from .data_sources import DataSource

class Axis(Enum):
    XAxis: int = 0
    YAxis: int = 1
    ZAxis: int = 2


class Measurement(ABC):
    def __init__(self, axis: Axis):
        super().__init__()
        if not isinstance(axis, Axis):
            raise ValueError(f"axis must be an Axis enum member (Axis.XAxis, "
                             f"Axis.YAxis, or Axis.ZAxis), got {axis!r}.")
        self.axis = axis

    @abstractmethod
    def compute_scale_factor(self, model: osim.Model, state: osim.State,
                             positions) -> float:
        pass


class FrameMeasurement(Measurement):
    def __init__(self, data_frame_1, data_frame_2, model_frame_path_1,
                 model_frame_path_2, axis: Axis):
        super().__init__(axis)
        self.data_frame_1 = data_frame_1
        self.data_frame_2 = data_frame_2
        self.model_frame_path_1 = model_frame_path_1
        self.model_frame_path_2 = model_frame_path_2

    def compute_scale_factor(self, model: osim.Model, state: osim.State,
                             positions) -> float:

        # Retrieve the model frames.
        model_frame_1 = osim.Frame.safeDownCast(
            model.getComponent(self.model_frame_path_1))
        model_frame_2 = osim.Frame.safeDownCast(
            model.getComponent(self.model_frame_path_2))

        # Magnitude of relative position between the two model frames.
        model_frame_1_position = model_frame_1.getPositionInGround(state).to_numpy()
        model_frame_2_position = model_frame_2.getPositionInGround(state).to_numpy()
        model_frame_distance = np.linalg.norm(model_frame_1_position -
                                              model_frame_2_position)

        # Magnitude of relative position between the data frames. Average over all
        # the frames in the data source.
        times = positions.getIndependentColumn()
        data_frame_distances = np.zeros(len(times))
        for i, t in enumerate(times):
            data_frame_1_position = positions.getDependentColumn(self.data_frame_1)[i]
            data_frame_2_position = positions.getDependentColumn(self.data_frame_2)[i]
            data_frame_distances[i] = np.linalg.norm(data_frame_1_position.to_numpy() -
                                                     data_frame_2_position.to_numpy())
        data_frame_distance = np.mean(data_frame_distances)

        # Compute the scale factor as the ratio between the data frame distance and the
        # model frame distance.
        scale_factor = data_frame_distance / model_frame_distance
        return scale_factor


class MarkerMeasurement(Measurement):
    def __init__(self, data_marker_1, data_marker_2, model_marker_path_1,
                 model_marker_path_2, axis: Axis):
        super().__init__(axis)
        self.data_marker_1 = data_marker_1
        self.data_marker_2 = data_marker_2
        self.model_marker_path_1 = model_marker_path_1
        self.model_marker_path_2 = model_marker_path_2

    def compute_scale_factor(self, model: osim.Model, state: osim.State,
                             positions) -> float:

        # Retrieve the model markers.
        model_marker_1 = osim.Marker.safeDownCast(
            model.getComponent(self.model_marker_path_1))
        model_marker_2 = osim.Marker.safeDownCast(
            model.getComponent(self.model_marker_path_2))

        # Magnitude of relative position between the two model markers.
        model_marker_1_position = model_marker_1.getLocationInGround(state).to_numpy()
        model_marker_2_position = model_marker_2.getLocationInGround(state).to_numpy()
        model_marker_distance = np.linalg.norm(model_marker_1_position -
                                               model_marker_2_position)

        # Magnitude of relative position between the data markers. Average over all
        # the frames in the data source.
        times = positions.getIndependentColumn()
        data_marker_distances = np.zeros(len(times))
        for i, t in enumerate(times):
            data_marker_1_position = positions.getDependentColumn(self.data_marker_1)[i]
            data_marker_2_position = positions.getDependentColumn(self.data_marker_2)[i]
            data_marker_distances[i] = np.linalg.norm(data_marker_1_position.to_numpy() -
                                                      data_marker_2_position.to_numpy())
        data_marker_distance = np.mean(data_marker_distances)

        # Compute the scale factor as the ratio between the data marker distance and the
        # model marker distance.
        scale_factor = data_marker_distance / model_marker_distance
        return scale_factor


class ModelScaler:
    def __init__(self, model: osim.Model, data_source: DataSource):
        self.model = model
        self.state = model.initSystem()
        self.data_source = data_source
        self.frame_map: dict[str, str] = {}
        self.marker_map: dict[str, str] = {}
        self.measurements: dict[str, list[Measurement]] = {}
        self.symmetry_pairs: list[tuple[str, str]] = []

    def register_frame(self, data_frame, model_frame):
        self.frame_map[data_frame] = model_frame

    def register_marker(self, data_marker, model_marker):
        self.marker_map[data_marker] = model_marker

    def add_measurement(self, segment, measurement: Measurement):
        if segment not in self.measurements:
            self.measurements[segment] = []
        self.measurements[segment].append(measurement)

    def add_frame_measurement(self, segment, frame_1, frame_2, axis: Axis):
        if frame_1 not in self.frame_map:
            raise KeyError(f"No model frame associated with data frame '{frame_1}' was "
                           "found. Please use register_frame() to associate a model "
                           "frame with this data frame before adding a measurement.")
        if frame_2 not in self.frame_map:
            raise KeyError(f"No model frame associated with data frame '{frame_2}' was "
                           "found. Please use register_frame() to associate a model "
                           "frame with this data frame before adding a measurement.")

        frame_path_1 = self.frame_map[frame_1]
        frame_path_2 = self.frame_map[frame_2]
        self.add_measurement(segment, FrameMeasurement(frame_1, frame_2,
                                                       frame_path_1, frame_path_2,
                                                       axis))

    def add_marker_measurement(self, segment, marker_1, marker_2, axis: Axis):
        if marker_1 not in self.marker_map:
            raise KeyError(f"No model marker associated with data marker '{marker_1}' "
                           "was found. Please use register_marker() to associate a "
                           "model marker with this data marker before adding a "
                           "measurement.")
        if marker_2 not in self.marker_map:
            raise KeyError(f"No model marker associated with data marker '{marker_2}' "
                           "was found. Please use register_marker() to associate a "
                           "model marker with this data marker before adding a "
                           "measurement.")

        marker_path_1 = self.marker_map[marker_1]
        marker_path_2 = self.marker_map[marker_2]
        self.add_measurement(segment, MarkerMeasurement(marker_1, marker_2,
                                                       marker_path_1, marker_path_2,
                                                       axis))

    def add_symmetry_pair(self, segment_1, segment_2):
        self.symmetry_pairs.append((segment_1, segment_2))

    def scale(self):

        # Import the C3D file and load the frame position data.
        positions = self.data_source.get_positions_table()

        # Create scale factors
        # --------------------
        scaleset = osim.ScaleSet()
        for segment, measurements in self.measurements.items():
            scale = self._create_scale(segment, measurements, positions)
            scaleset.cloneAndAppend(scale)

        # Apply symmetry to scale factors.
        # --------------------------------
        for segment_1, segment_2 in self.symmetry_pairs:
            scale_1 = scaleset.get(segment_1)
            scale_2 = scaleset.get(segment_2)
            factors_1 = scale_1.getScaleFactors()
            factors_2 = scale_2.getScaleFactors()
            avg_factors = osim.Vec3(
                0.5 * (factors_1[0] + factors_2[0]),
                0.5 * (factors_1[1] + factors_2[1]),
                0.5 * (factors_1[2] + factors_2[2]))
            scale_1.setScaleFactors(avg_factors)
            scale_2.setScaleFactors(avg_factors)

        # Scale the model
        # ---------------
        self.model.scale(self.state, scaleset, True)
        self.model.finalizeConnections()
        self.model.initSystem()

        return self.model

    def _create_scale(self, segment, measurements: list[Measurement], positions):
        scale = osim.Scale()
        scale.setSegmentName(segment)
        axis_factors = {Axis.XAxis: [], Axis.YAxis: [], Axis.ZAxis: []}

        for measurement in measurements:
            axis_factors[measurement.axis].append(measurement.compute_scale_factor(
                self.model, self.state, positions))

        factors = osim.Vec3(1.0)
        for axis, axis_measurements in axis_factors.items():
            if len(axis_measurements) > 0:
                avg_axis_factor = np.mean(axis_measurements)
                factors[axis.value] = avg_axis_factor

        scale.setScaleFactors(factors)

        return scale
