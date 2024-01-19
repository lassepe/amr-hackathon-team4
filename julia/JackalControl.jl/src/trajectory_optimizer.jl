function setup_trajectory_optimizer(;
    warmup = true,
    dt = 0.1,
    dynamics = UnicycleDynamics(; dt, control_bounds = (; lb = [-4.0, -1.4], ub = [4.0, 1.4])),
    planning_horizon = 15,
)
    minimum_px = -4.0
    maximum_px = 4.0
    minimum_py = -2.5
    maximum_py = 2.5
    maximum_lateral_acceleration = 0.75

    maximum_velocity = 1.5
    minimum_obstacle_distance = 0.75
    goal_dimension = 3
    obstacle_dimension = 3
    println("setting up...")
    state_dimension = state_dim(dynamics)
    control_dimension = control_dim(dynamics)
    primal_dimension = (state_dimension + control_dimension) * planning_horizon
    parameter_dimension = state_dimension + goal_dimension + obstacle_dimension

    unflatten_parameters = function (θ)
        θ_iter = Iterators.Stateful(θ)
        initial_state = first(θ_iter, state_dimension)
        goal = first(θ_iter, goal_dimension)
        obstacle = first(θ_iter, obstacle_dimension)
        (; initial_state, goal, obstacle)
    end

    function flatten_parameters(; initial_state, goal, obstacle)
        vcat(initial_state, goal, obstacle)
    end

    objective = function (z, θ)
        (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
        (; goal) = unflatten_parameters(θ)

        sum(u[1] .^ 2 + 2 * u[2] .^ 2 for u in us)
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
        function (z, θ)
            (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
            mapreduce(vcat, 2:length(xs)) do k
                (; obstacle) = unflatten_parameters(θ)
                px, py = xs[k]
                distance_to_obstacle_squared = (px - obstacle[1])^2 + (py - obstacle[2])^2
                [distance_to_obstacle_squared - minimum_obstacle_distance^2]
            end
        end,

        # limit acceleration and don't go too fast, stay within the playing field
        function (z, θ)
            (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
            mapreduce(vcat, 1:length(xs)) do k
                px, py, v, θ = xs[k]
                a, ω = us[k]

                lateral_acceleration = v * ω
                lateral_accelerartion_constraint = [
                    lateral_acceleration + maximum_lateral_acceleration,
                    -lateral_acceleration + maximum_lateral_acceleration,
                ]

                velocity_constraint = vcat(v + maximum_velocity, -v + maximum_velocity)
                position_constraints =
                    vcat(px - minimum_px, maximum_px - px, py - minimum_py, maximum_py - py)
                vcat(lateral_accelerartion_constraint, velocity_constraint, position_constraints)
            end
        end,

        # reach the goal position
        function (z, θ)
            (; xs) = unflatten_trajectory(z, state_dimension, control_dimension)
            (; goal, obstacle) = unflatten_parameters(θ)
            # can also track the obstacle instead
            #goal_position_deviation = xs[end][1:2] .- obstacle[1:2]
            goal_position_deviation = xs[end][1:2] .- goal[1:2]
            [
                goal_position_deviation .+ 0.1
                -goal_position_deviation .+ 0.1
            ]
        end,

        ## reach goal orientation
        # this gave rather jerky behavior; removing for now
        #function (z, θ)
        #    (; xs) = unflatten_trajectory(z, state_dimension, control_dimension)
        #    (; goal) = unflatten_parameters(θ)
        #    goal_orientation_deviation = xs[end][4] - goal[3]
        #    [
        #        goal_orientation_deviation .+ 0.01
        #        -goal_orientation_deviation .+ 0.01
        #    ]
        #end,
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

    function trajectory_optimizer(initial_state, goal, obstacle)
        θ = flatten_parameters(; initial_state, goal, obstacle)
        solution = OrderedPreferences.solve(problem, θ; warmstart_solution = solution)
        unflatten_trajectory(solution.primals, state_dim(dynamics), control_dim(dynamics))
    end

    # trigger compilation of the full stack
    if warmup
        println("warming up...")
        trajectory_optimizer(
            zeros(state_dimension),
            zeros(obstacle_dimension),
            zeros(goal_dimension),
        )
        println("done warming up...")
    end

    trajectory_optimizer
end
