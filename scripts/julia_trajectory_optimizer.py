from juliacall import Main as jl
from utils import OpenLoopStrategy


class JuliaTrajectoryOptimizer:
    def __init__(self):
        self.jl = jl
        self.jl.seval("using Revise: Revise")
        self.jl.seval("using JackalControl: JackalControl")
        self.julia_solver = self.jl.JackalControl.setup_trajectory_optimizer()

    def compute_strategy(self, initial_state, goal, obstacle):
        optimized_trajectory = self.julia_solver(initial_state, goal, obstacle)
        return self._get_strategy_from_trajectory(optimized_trajectory)

    def _get_strategy_from_trajectory(self, trajectory):
        """
        Extract the control inputs from the optimized trajectory.

        The robot expects longitudinal velocity (v) and turn rate (\omega) as inputs but the
        trajectory has states [px, py, v, \theta] and inputs [a, \omega].
        """
        us = []
        for i in range(1, len(trajectory.xs)):
            velocity = trajectory.xs[i][2]
            turn_rate = trajectory.us[i - 1][1]
            us.append([velocity, turn_rate])

        return OpenLoopStrategy(us)
