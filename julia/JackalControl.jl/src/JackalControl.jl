module JackalControl

using TrajectoryGamesExamples: UnicycleDynamics
using TrajectoryGamesBase: unflatten_trajectory, state_dim, control_dim, control_bounds
using OrderedPreferences: OrderedPreferences, ParametricOrderedPreferencesProblem

include("trajectory_optimizer.jl")
include("Server.jl")

end # module JackalControl
