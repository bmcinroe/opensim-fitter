import os
from xml.parsers.expat import model
import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from .utilities import get_coordinate_indexes, get_ipopt_options
from .callbacks import TrackingCostCallback, TrackingCostJacobianCallback


class Solver(ABC):

    @abstractmethod
    def solve(self) -> osim.TimeSeriesTable:
        pass


# weights = {'position': 2.0,
#            'orientation': 5.0,
#            'smoothness': 0.5}

class InverseKinematicsSolver(Solver):

    def __init__(self, model, positions, orientations, convergence_tolerance=1e-6,
                 finite_differences=False, position_weight=1.0, orientation_weight=1.0,
                 smoothness_weight=0.01):

        # Load the model.
        modelProcessor = osim.ModelProcessor(model)
        modelProcessor.append(osim.ModOpRemoveMuscles())
        self.model =  modelProcessor.process()
        self.state = self.model.initSystem()
        # For now, disallow models with joints where qdot != u.
        assert(self.state.getNQ() == self.state.getNU())

        # Create a mapping between coordinate paths and their indexes in the state
        # vector.
        self.coordinates_map = get_coordinate_indexes(self.model,
                                                      skip_dependent_coordinates=True)
        self.coordinate_indexes = list(self.coordinates_map.values())

        # Tracking data.
        self.positions = positions
        self.orientations = orientations

        # Optimization settings.
        self.convergence_tolerance = convergence_tolerance
        self.finite_differences = finite_differences
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.smoothness_weight = smoothness_weight

    def run_optimization(self, frame_paths, positions, quaternions,
                     weights, x0, lbx, ubx):
        # Declare optimization variables.
        x = ca.SX.sym('x', len(self.coordinate_indexes))

        # Construct the callback function defining the tracking cost.
        # If 'finite_differences' is True, the Jacobian will be computed using finite
        # differences. Otherwise, a callback function that provides an analytical
        # Jacobian will be used.
        if self.finite_differences:
            track = TrackingCostCallback('tracking_cost', self.model,
                                         self.coordinate_indexes,
                                         frame_paths, positions, quaternions,
                                         weights, {'enable_fd': True})
        else:
            track = TrackingCostJacobianCallback('tracking_cost', self.model,
                                                 self.coordinate_indexes,
                                                 frame_paths, positions, quaternions,
                                                 weights)
        tracking_cost = ca.Function('f', [x], [track(x)])

        # The total cost function includes a smoothness term to penalize large
        # deviations from the previous solution.
        f = tracking_cost(x) + weights['smoothness'] * ca.sumsqr(x - x0)

        # Form the non-linear program (NLP).
        nlp = {'x': x, 'f': f}

        # Allocate a solver.
        opts = {}
        opts['ipopt'] = get_ipopt_options(self.convergence_tolerance)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Solve the NLP.
        sol = solver(x0=x0, lbx=lbx, ubx=ubx)
        return sol

    def solve(self) -> osim.TimeSeriesTable:

        # Load tracking data
        # ------------------
        frame_paths = self.positions.getColumnLabels()
        times = self.positions.getIndependentColumn()

        # Inverse kinematics
        # ------------------
        # Define initial guess and bounds.
        # This utility retrieves a mapping between coordinate paths and their indexes in the
        # state vector.
        x0 = []
        lbx = []
        ubx = []
        for coord_path, ix in self.coordinates_map.items():
            coord = osim.Coordinate.safeDownCast(self.model.getComponent(coord_path))
            x0.append(coord.getDefaultValue())
            lbx.append(coord.getRangeMin())
            ubx.append(coord.getRangeMax())

        # Solve position-only optimization to create an inital guess for the full IK
        # problem.
        print('Solving initial guess optimization...')
        sol = self.run_optimization(frame_paths,
                                    self.positions.getRowAtIndex(0),
                                    self.orientations.getRowAtIndex(0),
                                    {'position': 10.0*self.position_weight,
                                     'orientation': 0.1*self.orientation_weight,
                                     'smoothness': 0.01*self.smoothness_weight},
                                    x0, lbx, ubx)
        x0 = sol['x']

        # Iterate over all of the time steps in the tracking data and solve the
        # optimization problem at each time step.
        statesTraj = osim.StatesTrajectory()
        for itime, time in enumerate(times):
            print(f'Solving time {itime+1} of {len(times)} (t={time:.3f} s)...')

            # Construct the callback function defining the tracking cost.
            sol = self.run_optimization(frame_paths,
                                        self.positions.getRowAtIndex(itime),
                                        self.orientations.getRowAtIndex(itime),
                                        {'position': self.position_weight,
                                         'orientation': self.orientation_weight,
                                         'smoothness': self.smoothness_weight},
                                        x0, lbx, ubx)

            # Save solution
            state = self.model.initSystem()
            state.setTime(time)
            q = np.zeros(state.getNQ())
            q[self.coordinate_indexes] = np.squeeze(sol['x'].full())
            state.setQ(osim.Vector.createFromMat(q))
            statesTraj.append(state)

            # Use the solution for the current time step as the initial guess for the next
            # time step.
            x0 = sol['x']

        # Export the solution to a .sto file.
        statesTable = statesTraj.exportToTable(self.model)

        return statesTable
