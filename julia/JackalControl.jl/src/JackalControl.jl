module JackalControl

using TrajectoryGamesExamples: UnicycleDynamics
using TrajectoryGamesBase: unflatten_trajectory, state_dim, control_dim, control_bounds
using OrderedPreferences: OrderedPreferences, ParametricOrderedPreferencesProblem

using PythonCall: PythonCall

include("pyutils.jl")

function setup_trajectory_optimizer()
    function trajectory_optimizer(initial_state, goal_position, obstacle_position)
        @pygil_unlocked begin
            (;
                xs = [[0, 0, 1.0, 0.0], [0, 0, 1.0, 0.0], [0, 0, 1.0, 0.0]],
                us = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            )
        end
    end
end

end # module JackalControl
