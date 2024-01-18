function setup_trajectory_optimizer(;
    warmup = true,
    dynamics = UnicycleDynamics(; control_bounds = (; lb = [-4.0, -1.4], ub = [4.0, 1.4])),
    planning_horizon = 10,
)
    goal_dimension = 2
    obstacle_dimension = 2
    println("setting up...")
    state_dimension = state_dim(dynamics)
    control_dimension = control_dim(dynamics)
    primal_dimension = (state_dimension + control_dimension) * planning_horizon
    parameter_dimension = state_dimension + goal_dimension + obstacle_dimension

    unflatten_parameters = function (θ)
        θ_iter = Iterators.Stateful(θ)
        initial_state = first(θ_iter, state_dimension)
        goal_position = first(θ_iter, 2)
        obstacle_position = first(θ_iter, 2)
        (; initial_state, goal_position, obstacle_position)
    end

    function flatten_parameters(; initial_state, goal_position, obstacle_position)
        vcat(initial_state, goal_position, obstacle_position)
    end

    objective = function (z, θ)
        (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
        (; goal_position) = unflatten_parameters(θ)

        sum(sum(u .^ 2) for u in us)
    end

    equality_constraints = function (z, θ)
        (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
        (; initial_state) = unflatten_parameters(θ)
        initial_state_constraint = xs[1] - initial_state
        dynamics_constraints = mapreduce(vcat, 2:length(xs)) do k
            xs[k] - dynamics(xs[k - 1], us[k - 1], k)
        end
        vcat(initial_state_constraint, dynamics_constraints)
    end

    function inequality_constraints(z, θ)
        (; lb, ub) = control_bounds(dynamics)
        lb_mask = findall(!isinf, lb)
        ub_mask = findall(!isinf, ub)
        (; us) = unflatten_trajectory(z, state_dimension, control_dimension)
        mapreduce(vcat, us) do u
            vcat(u[lb_mask] - lb[lb_mask], ub[ub_mask] - u[ub_mask])
        end
    end

    prioritized_inequality_constraints = [
        # limit acceleration and don't go too fast, stay within the playing field
        function (z, θ)
            (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
            mapreduce(vcat, 1:length(xs)) do k
                px, py, v, θ = xs[k]
                a, ω = us[k]

                lateral_acceleration = v * ω
                lateral_accelerartion_constraint =
                    [lateral_acceleration + 1.0, -lateral_acceleration + 1.0]

                velocity_constraint = vcat(v + 2.0, -v + 2.0)
                position_constraints = vcat(px + 2.0, -px + 2.0, py + 2.0, -py + 2.0)
                vcat(lateral_accelerartion_constraint, velocity_constraint, position_constraints)
            end
        end,

        # reach the goal
        function (z, θ)
            (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
            (; goal_position) = unflatten_parameters(θ)
            goal_deviation = xs[end][1:2] .- goal_position
            [
                goal_deviation .+ 0.1
                -goal_deviation .+ 0.1
            ]
        end,
    ]

    problem = ParametricOrderedPreferencesProblem(;
        objective,
        equality_constraints,
        inequality_constraints,
        prioritized_inequality_constraints,
        primal_dimension,
        parameter_dimension,
    )

    println("done setting up...")

    (; problem, flatten_parameters, unflatten_parameters)
    solution = nothing

    function trajectory_optimizer(initial_state, goal_position, obstacle_position)
        θ = flatten_parameters(; initial_state, goal_position, obstacle_position)
        solution = OrderedPreferences.solve(problem, θ; warmstart_solution = solution)
        unflatten_trajectory(solution.primals, state_dim(dynamics), control_dim(dynamics))
    end

    # trigger compilation of the full stack
    if warmup
        println("warming up...")
        trajectory_optimizer(zeros(state_dimension), zeros(2), zeros(2))
        println("done warming up...")
    end

    trajectory_optimizer
end
