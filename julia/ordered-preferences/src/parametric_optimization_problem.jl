""" Practice: write a standard-form optimization problem as a MCP.
i.e., take a problem of the form
             min_{x ∈ Rⁿ} f(x, θ)
             s.t.            g(x, θ) = 0
                             h(x, θ) ≥ 0

where θ is a vector of parameters. Express it in the following form:
             find   z
             s.t.   F(z, θ) ⟂ z̲ ≤ z ≤ z̅
where we interpret z = (x, λ, μ), with λ and μ the Lagrange multipliers
for the constraints g and h, respectively. The expression F(z) ⟂ z̲ ≤ z ≤ z̅
should be read as the following three statements:
            - if z = z̲, then F(z, θ) ≥ 0
            - if z̲ < z < z̅, then F(z, θ) = 0
            - if z = z̅, then F(z, θ) ≤ 0

For more details, please consult the documentation for the package
`Complementarity.jl`, which may be found here:
https://github.com/chkwon/Complementarity.jl/tree/master
"""

"Generic description of a constrained optimization problem."
struct ParametricOptimizationProblem{T1,T2,T3,T4<:ParametricMCP}
    "Objective function"
    objective::T1
    "Equality constraint"
    equality_constraint::T2
    "Inequality constraint"
    inequality_constraint::T3
    "Dimension of parameter vector"
    parameter_dimension::Int
    "Dimension of primal variable"
    primal_dimension::Int
    "Dimension of equality constraint"
    equality_dimension::Int
    "Dimension of inequality constraint"
    inequality_dimension::Int
    "Corresponding parametric MCP"
    parametric_mcp::T4
end

function ParametricOptimizationProblem(;
    objective,
    equality_constraint,
    inequality_constraint,
    parameter_dimension,
    primal_dimension,
    equality_dimension,
    inequality_dimension,
    compute_sensitivities = false,
)
    @assert !isnothing(equality_constraint)
    @assert !isnothing(inequality_constraint)

    total_dimension = primal_dimension + equality_dimension + inequality_dimension

    # Define symbolic variables for this MCP.
    z̃ = Symbolics.scalarize(only(Symbolics.@variables(z̃[1:total_dimension])))
    z = BlockArray(z̃, [primal_dimension, equality_dimension, inequality_dimension])

    x = z[Block(1)]
    λ = z[Block(2)]
    μ = z[Block(3)]

    # Define a symbolic variable for the parameters.
    Symbolics.@variables θ̃[1:(parameter_dimension)]
    θ = Symbolics.scalarize(θ̃)
    if isempty(θ)
        θ = Symbolics.Num[]
    end

    # Build symbolic expressions for objective and constraints.
    f = objective(x, θ)
    g = equality_constraint(x, θ)
    if isempty(g)
        g = Symbolics.Num[]
    end

    h = inequality_constraint(x, θ)
    if isempty(h)
        h = Symbolics.Num[]
    end

    # Build Lagrangian.
    L = f - λ' * g - μ' * h

    # Build F = [∇ₓL, g, h]'.
    ∇ₓL = Symbolics.gradient(L, x)
    F = Vector{Symbolics.Num}([∇ₓL; g; h])

    # Set lower and upper bounds for z.
    z̲ = [
        fill(-Inf, primal_dimension)
        fill(-Inf, equality_dimension)
        fill(0, inequality_dimension)
    ]
    z̅ = [
        fill(Inf, primal_dimension)
        fill(Inf, equality_dimension)
        fill(Inf, inequality_dimension)
    ]

    # Build parametric MCP.
    parametric_mcp = ParametricMCP(F, z̃, θ, z̲, z̅; compute_sensitivities)

    ParametricOptimizationProblem(
        objective,
        equality_constraint,
        inequality_constraint,
        parameter_dimension,
        primal_dimension,
        equality_dimension,
        inequality_dimension,
        parametric_mcp,
    )
end

function total_dim(problem::ParametricOptimizationProblem)
    problem.primal_dimension + problem.equality_dimension + problem.inequality_dimension
end

"Solve a constrained parametric optimization problem."
function solve(
    problem::ParametricOptimizationProblem,
    parameter_value = zeros(problem.parameter_dimension);
    initial_guess = nothing,
    verbose = false,
)
    z0 = if !isnothing(initial_guess)
        initial_guess
    else
        zeros(total_dim(problem))
    end

    z, status, info = ParametricMCPs.solve(
        problem.parametric_mcp,
        parameter_value;
        initial_guess = z0,
        verbose,
        cumulative_iteration_limit = 100000,
        proximal_perturbation = 1e-2,
        use_basics = true,
        use_start = true,
    )

    primals = z[1:(problem.primal_dimension)]

    (; primals, variables = z, status, info)
end
