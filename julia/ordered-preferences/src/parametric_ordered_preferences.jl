struct ParametricOrderedPreferencesProblem{T1<:Vector{<:ParametricOptimizationProblem}}
    subproblems::T1
end

"""
Synthesizes a parametric ordered preferences problem from user functions
"""
function ParametricOrderedPreferencesProblem(;
    objective,
    equality_constraints,
    inequality_constraints,
    prioritized_inequality_constraints,
    primal_dimension,
    parameter_dimension,
)
    # Problem data
    ordered_priority_levels = eachindex(prioritized_inequality_constraints)

    dummy_primals = zeros(primal_dimension)
    dummy_parameters = zeros(parameter_dimension)

    equality_dimension = length(equality_constraints(dummy_primals, dummy_parameters))
    inequality_dimension = length(inequality_constraints(dummy_primals, dummy_parameters))

    fixed_slack_dimension = 0
    inner_inequality_constraints = Any[inequality_constraints]

    subproblems = (ParametricOptimizationProblem[])

    function set_up_level(priority_level)
        parameter_dimension_ii = parameter_dimension + fixed_slack_dimension

        if isnothing(priority_level)
            # the final level does not have any additional slacks
            slack_dimension_ii = 0
        else
            prioritized_constraints_ii = prioritized_inequality_constraints[priority_level]
            slack_dimension_ii = length(prioritized_constraints_ii(dummy_primals, dummy_parameters))
        end

        primal_dimension_ii = primal_dimension + slack_dimension_ii

        if isnothing(priority_level)
            objective_ii = objective
        else
            objective_ii = function (x, θ)
                # everything beyond the original primal dimension are the slacks for this level
                sum(x[(primal_dimension + 1):end] .^ 2)
            end
        end

        equality_constraints_ii = function (x, θ)
            # only forward the original primals and parameters
            x_original = x[1:primal_dimension]
            θ_original = θ[1:parameter_dimension]
            equality_constraints(x_original, θ_original)
        end

        inequality_constraints_ii = function (x, θ)
            original_x = x[1:primal_dimension]
            original_θ = θ[1:parameter_dimension]
            fixed_slacks = θ[(parameter_dimension + 1):end]
            @assert length(fixed_slacks) == fixed_slack_dimension
            slacks_ii = x[(primal_dimension + 1):end]
            @assert length(slacks_ii) == slack_dimension_ii

            unslacked_constraints =
                mapreduce(vcat, inner_inequality_constraints) do constraint
                    constraint(original_x, original_θ)
                end + vcat(zeros(inequality_dimension), fixed_slacks)

            if isnothing(priority_level)
                return unslacked_constraints
            end

            vcat(
                unslacked_constraints,
                prioritized_constraints_ii(original_x, original_θ) .+ slacks_ii,
            )
        end

        inequality_dimension_ii = let
            dummy_primals_ii = zeros(primal_dimension_ii)
            dummy_parameter_ii = zeros(parameter_dimension_ii)
            length(inequality_constraints_ii(dummy_primals_ii, dummy_parameter_ii))
        end

        optimization_problem = ParametricOptimizationProblem(;
            objective = objective_ii,
            equality_constraint = equality_constraints_ii,
            inequality_constraint = inequality_constraints_ii,
            parameter_dimension = parameter_dimension_ii,
            primal_dimension = primal_dimension_ii,
            equality_dimension = equality_dimension,
            inequality_dimension = inequality_dimension_ii,
        )

        fixed_slack_dimension += slack_dimension_ii
        if !isnothing(priority_level)
            push!(inner_inequality_constraints, prioritized_inequality_constraints[priority_level])
        end
        push!(subproblems, optimization_problem)
    end

    for priority_level in ordered_priority_levels
        set_up_level(priority_level)
    end
    set_up_level(nothing)

    ParametricOrderedPreferencesProblem(subproblems)
end

# TODO: allow for user-defined warm-starting
function solve(
    ordered_preferences_problem::ParametricOrderedPreferencesProblem,
    θ = Float64[];
    warmstart_solution = nothing,
    extra_slack = 1e-4, # TODO: could also express this as inner tightening
    warmstart_strategy = :cascade,
)
    outermost_problem = last(ordered_preferences_problem.subproblems)

    fixed_slacks = Float64[]

    # TODO: optimize allocation and type stability
    if isnothing(warmstart_solution)
        warmstart_primals = nothing
        warmstart_slacks = nothing
    else
        warmstart_primals = warmstart_solution.primals
        warmstart_slacks = warmstart_solution.slacks
    end

    level_solutions = []

    for (level, optimization_problem) in enumerate(ordered_preferences_problem.subproblems)
        initial_guess = zeros(total_dim(optimization_problem))
        if !isnothing(warmstart_solution)
            if warmstart_strategy === :cascade || warmstart_strategy === :final
                # concatenate the outer primals (appearing in all subproblems) with the slacks for this
                # level
                slack_dimension_ii =
                    optimization_problem.primal_dimension - outermost_problem.primal_dimension
                if !isnothing(warmstart_slacks)
                    fixed_slack_dimension = length(fixed_slacks)
                    slacks_ii_warmstart =
                        warmstart_slacks[(fixed_slack_dimension + 1):(fixed_slack_dimension + slack_dimension_ii)]
                else
                    slacks_ii_warmstart = zeros(slack_dimension_ii)
                end

                if warmstart_strategy === :final
                    warmstart_primals = warmstart_solution.level_solutions[end].primals
                end

                initial_guess[1:(optimization_problem.primal_dimension)] .= begin
                    vcat(warmstart_primals[1:(outermost_problem.primal_dimension)], slacks_ii_warmstart)
                end
            elseif warmstart_strategy === :parallel
                _wp = warmstart_solution.level_solutions[level].primals
                initial_guess[1:length(_wp)] .= _wp
            else
                error("invalid warmstart strategy")
            end
        end

        parameter_value = vcat(θ, fixed_slacks)
        solution = solve(optimization_problem, parameter_value; initial_guess)
        append!(
            fixed_slacks,
            solution.primals[(outermost_problem.primal_dimension + 1):end] .+ extra_slack,
        )
        warmstart_primals = solution.primals
        push!(level_solutions, solution)
    end

    # TODO: would be nice if the user can still associate the slacks with the corresponding constraints
    (; level_solutions[end]..., slacks = fixed_slacks, level_solutions)
end
