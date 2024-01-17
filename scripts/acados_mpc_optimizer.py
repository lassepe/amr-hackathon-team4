#!/usr/bin/env python

from acados_template import AcadosSimSolver, AcadosOcpSolver, AcadosSim, AcadosOcp, AcadosModel
from casadi import SX, vertcat, sin, cos
import numpy as np
import time
import matplotlib.pyplot as plt

def unicycle_ode_model() -> AcadosModel:
    # returns a AcadosModel object containing the symbolic representation

    model_name = 'unicycle'

    # constants

    # set up states & controls
    x1      = SX.sym('x1')
    y1      = SX.sym('y1')
    theta   = SX.sym('theta')
    x = vertcat(x1, y1, theta)

    v = SX.sym('v')
    w = SX.sym('w')
    u = vertcat(v,w)

    # xdot
    x1_dot      = SX.sym('x1_dot')
    y1_dot      = SX.sym('y1_dot')
    theta_dot   = SX.sym('theta_dot')
    xdot = vertcat(x1_dot, y1_dot, theta_dot)

    # dynamics
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    f_expl = vertcat(v*cos_theta, v*sin_theta, w)

    # Coverting to AcadosModel
    model = AcadosModel()
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model


class MPC:
    def __init__(self, model_acados, N, t_horizon, x_goal):
        self.N = N # prediction horizon
        self.model = model_acados
        self.t_horizon = t_horizon
        self.x_goal = x_goal

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())

    def ocp(self):
        model = self.model

        t_horizon = self.t_horizon
        N = self.N

        # Get model
        model_ac = model
        model_ac.p = model.p

        # Dimensions: TODO: set the dimensions in the model
        nx = model.x.size()[0] # number of states
        nu = model.u.size()[0] # number of control inputs
        # ny = nx # number of ? nx+nu

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N 
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        # ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        # Initialize cost function. info can be found here: https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf
        # for quadratic cost example see: https://github.com/TUM-AAS/ml-casadi/blob/44ec47f0d14aa8306d0f5dc786babddfbc8964f9/examples/mpc_mlp_naive_example.py#L137

        # the 'EXTERNAL' cost type can be used to define general cost terms
        # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.

        Q_mat = 2*np.diag([0, 0, 0])
        R_mat = 2*np.diag([1e-2, 1e-2])
        Qe_mat = 2*np.diag([100, 100, 0])
        x_goal = self.x_goal.reshape(-1, 1)

        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost = model.x.T @ Q_mat @ model.x + model.u.T @ R_mat @ model.u
        ocp.model.cost_expr_ext_cost_e = (model.x - x_goal).T @ Qe_mat @ (model.x - x_goal)

        # Initial reference trajectory (will be overwritten)
        # ocp.cost.yref = np.zeros(1)

        # Initial state (will be overwritten)
        ocp.constraints.x0 = np.zeros(nx)

        # Set constraints
        v_forward_max = 0.5
        v_backwards_max = 0.1
        w_max = 1.0
        ocp.constraints.lbu = np.array([-v_backwards_max, -w_max])
        ocp.constraints.ubu = np.array([v_forward_max, w_max])
        ocp.constraints.idxbu = np.arange(nu) # all elements of u

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.integrator_type = 'ERK' # which means explicit RK4, can also be 'IRK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP

        # ocp.parameter_values = model.parameter_values

        return ocp

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.name = model.name
        return model_ac

def run_as_fast(run_many_times=100):
    N = 10 # discretization steps
    t_horizon = 5.
    x_goal = np.array([-5., -1., 0.])

    model = unicycle_ode_model()
    solver = MPC(model_acados=model, N=N, t_horizon=t_horizon, x_goal=x_goal).solver

    xt = np.array([0., 0., 0.])
    x = np.zeros((model.x.size()[0], run_many_times))
    x[:,0] = xt
    u = np.zeros((model.u.size()[0], run_many_times))

    ts = t_horizon / N
    opt_times = []

    for i in range(run_many_times):
        now = time.time()
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)
        solver.solve()
        xt = solver.get(1, "x")
        ut = solver.get(0, "u")
        x[:,i] = xt
        u[:,i] = ut

        # x_l = []
        # for i in range(N):
        #     x_l.append(solver.get(i, "x"))

        elapsed = time.time() - now
        opt_times.append(elapsed)

    print(f'Mean iteration time : {1000*np.mean(opt_times):.2f}ms -- {1/np.mean(opt_times):.0f}Hz)')
    plot_trajectory(x, u)


def plot_trajectory(x, u):
    plt.plot(x[0,:], x[1,:], 'o', label='Trajectory')
    plt.plot(x[0,0], x[1,0], 'ro', label='Initial Point')
    
    # Add arrows pointing in the direction of theta
    plt.quiver(x[0,:-1], x[1,:-1], np.cos(x[2,:-1]), np.sin(x[2,:-1]), angles='xy', scale_units='xy', scale=5, width=0.005)
    
    plt.grid(True)  # Add grid
    plt.axis("equal")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run_as_fast()
