module OrderedPreferences

using LinearAlgebra

using ParametricMCPs: ParametricMCPs, ParametricMCP
using Symbolics: Symbolics
using BlockArrays: BlockArrays, BlockArray, Block

include("parametric_optimization_problem.jl")
export ParametricOptimizationProblem, solve, total_dim

include("parametric_ordered_preferences.jl")
export ParametricOrderedPreferencesProblem, solve

end # module OrderedPreferences
