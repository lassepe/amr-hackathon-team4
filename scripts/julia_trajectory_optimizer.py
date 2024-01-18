from utils import OpenLoopStrategy

import json
from websockets.sync.client import connect


class JuliaTrajectoryOptimizer:
    def __init__(self, server_address="ws://127.0.0.1:8081"):
        self.websocket = connect(server_address)

    # make sure we close the websocket when we're done
    def __del__(self):
        self.websocket.close()

    def compute_strategy(self, state, goal, obstacle):
        request = json.dumps(
            {
                "state": state,
                "goal": goal,
                "obstacle": obstacle,
            }
        )
        self.websocket.send(request)
        response = self.websocket.recv()
        trajectory = json.loads(response)
        return self._get_strategy_from_trajectory(trajectory)

    def _get_strategy_from_trajectory(self, trajectory):
        """
        Extract the control inputs from the optimized trajectory.

        The robot expects longitudinal velocity (v) and turn rate (\omega) as inputs but the
        trajectory has states [px, py, v, \theta] and inputs [a, \omega].
        """
        control_inputs = []

        traj_xs = trajectory["xs"]
        traj_us = trajectory["us"]
        for i in range(1, len(traj_xs)):
            velocity = (traj_xs[i][2] + traj_xs[i - 1][2]) / 2
            turn_rate = traj_us[i - 1][1]
            control_inputs.append([velocity, turn_rate])

        return OpenLoopStrategy(control_inputs)
