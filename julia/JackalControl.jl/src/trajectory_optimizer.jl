function setup_trajectory_optimizer()
    function trajectory_optimizer(initial_state, goal_position, obstacle_position)
        (;
            xs = [[0, 0, 1.0, 0.0], [0, 0, 1.0, 0.0], [0, 0, 1.0, 0.0]],
            us = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        )
    end
end
