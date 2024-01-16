using OrderedPreferences
using Test: @testset, @test

include("../experiments/SimpleLinearExample.jl")

@testset "All tests" begin
    @testset "OptimizationTests" begin
        f(x, θ) = sum(x)
        g(x, θ) = [sum(x .^ 2) - 1]
        h(x, θ) = -x

        problem = ParametricOptimizationProblem(;
            objective = f,
            equality_constraint = g,
            inequality_constraint = h,
            parameter_dimension = 1,
            primal_dimension = 2,
            equality_dimension = 1,
            inequality_dimension = 2,
        )

        (; primals, variables, status, info) = solve(problem)
        println("primals: ", primals)
        println("variables: ", variables)
        println("status: ", status)
        println("info: ", info)
        @test all(isapprox.(primals, -0.5sqrt(2), atol = 1e-6))
    end

    @testset "Simple linear ordered preferences problem" begin
        ordered_preferences_problem = SimpleLinearExample.get_problem()
        solution = solve(ordered_preferences_problem)
        @test isapprox(solution.primals, [6, 5])
    end
end
