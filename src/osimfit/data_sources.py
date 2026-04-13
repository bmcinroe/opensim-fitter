import ezc3d
import numpy as np
import opensim as osim
from abc import ABC, abstractmethod

class DataSource(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_positions_table(self):
        pass

    @abstractmethod
    def get_quaternions_table(self):
        pass

class C3DSource(DataSource):
    def __init__(self, filepath, columns_to_ignore=[], label_map={}):
        super().__init__()

        self.filepath = filepath
        self.c3d = ezc3d.c3d(filepath)

        # This is a Y-Z space-fixed rotation needed to convert data collected from Theia
        # to OpenSim's ground reference frame convention (X forward, Y up, Z right).
        osim_rotation = osim.Rotation()
        osim_rotation.setRotationFromTwoAnglesTwoAxes(1, # space-fixed
                -0.5*np.pi, osim.CoordinateAxis(1), # Y rotation
                -0.5*np.pi, osim.CoordinateAxis(2)) # Z rotation
        self.osim_rotation = osim_rotation

        # This is an additional body-fixed rotation that effectively swaps the axes of
        # the rotations collected from Theia to match OpenSim's ground reference frame
        # convention (X forward, Y up, Z right), which is the convention used by the
        # matching Frame elements in the generic model.
        frame_rotation = osim.Rotation()
        frame_rotation.setRotationToBodyFixedXY(osim.Vec2(0.5*np.pi))
        self.frame_rotation = frame_rotation

        # Columns that should be ignored when processing the data.
        self.columns_to_ignore = columns_to_ignore

        # Map of original labels to new labels.
        self.label_map = label_map

    def get_positions_table(self):
        data = self._get_data('rotations')
        num_frames = data.shape[3]
        labels = self._get_data_labels('ROTATION')
        rate = self._get_data_rate('ROTATION')
        times = self._get_time_vector(rate, num_frames)

        table = osim.TimeSeriesTableVec3()
        for iframe in range(num_frames):
            row = osim.RowVectorVec3(len(labels), osim.Vec3(0))
            for ilabel, label in enumerate(labels):
                position = data[:, 3, ilabel, iframe] / 1000.0  # mm to m
                row[ilabel] = osim.Vec3(position[0], position[1], position[2])
                row[ilabel] = self.osim_rotation.multiply(row[ilabel])

            table.appendRow(times[iframe], row)

        table.setColumnLabels(labels)
        table = self._remove_ignored_columns(table)
        table = self._update_column_labels(table)
        table.addTableMetaDataString("Units", "m")
        table.addTableMetaDataString("DataRate", str(rate))

        return table

    def get_quaternions_table(self):
        data = self._get_data('rotations')
        num_frames = data.shape[3]
        labels = self._get_data_labels('ROTATION')
        rate = self._get_data_rate('ROTATION')
        times = self._get_time_vector(rate, num_frames)

        table = osim.TimeSeriesTableQuaternion()
        for iframe in range(num_frames):
            row = osim.RowVectorQuaternion(len(labels), osim.Quaternion())
            for ilabel, label in enumerate(labels):
                rot = data[:3, :3, ilabel, iframe]
                data_rotation = osim.Rotation()
                data_rotation.set(0,0, rot[0,0])
                data_rotation.set(1,0, rot[1,0])
                data_rotation.set(2,0, rot[2,0])
                data_rotation.set(0,1, rot[0,1])
                data_rotation.set(1,1, rot[1,1])
                data_rotation.set(2,1, rot[2,1])
                data_rotation.set(0,2, rot[0,2])
                data_rotation.set(1,2, rot[1,2])
                data_rotation.set(2,2, rot[2,2])
                rotation = self.osim_rotation.multiply(data_rotation)
                rotation = rotation.multiply(self.frame_rotation)

                # Store as a quaternion.
                new_quat = rotation.convertRotationToQuaternion()
                upd_quat = row.updElt(0, ilabel)
                upd_quat.set(0, new_quat.get(0))
                upd_quat.set(1, new_quat.get(1))
                upd_quat.set(2, new_quat.get(2))
                upd_quat.set(3, new_quat.get(3))

            table.appendRow(times[iframe], row)

        table.setColumnLabels(labels)
        table = self._remove_ignored_columns(table)
        table = self._update_column_labels(table)
        table.addTableMetaDataString("DataRate", str(rate))

        return table

    def _get_data(self, parameter):
        return self.c3d.data[parameter]

    def _get_data_labels(self, parameter):
        raw_labels = self.c3d.parameters[parameter]['LABELS']['value']
        labels = [label.replace('_4X4', '') for label in raw_labels]
        return labels

    def _get_data_rate(self, parameter):
        return self.c3d.parameters[parameter]['RATE']['value'][0]

    def _get_time_vector(self, rate, num_frames):
        return np.array([i/rate for i in range(num_frames)])

    def _remove_ignored_columns(self, table):
        for col in self.columns_to_ignore:
            table.removeColumn(col)
        return table

    def _update_column_labels(self, table):
        if not self.label_map:
            return table

        labels = list(table.getColumnLabels())
        for ilabel in range(len(labels)):
            labels[ilabel] = self.label_map.get(labels[ilabel])
        table.setColumnLabels(labels)
        return table

