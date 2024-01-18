from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
from casadi import SX, MX, vertcat, sin, cos
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import OpenLoopStrategy

def unicycle_ode_model() -> AcadosModel:
    # returns a AcadosModel object containing the symbolic representation

    model_name = 'unicycle'

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

    # parameters: end goal
    x1_e      = SX.sym('x1_e')
    y1_e      = SX.sym('y1_e')
    theta_e   = SX.sym('theta_e')
    x_goal = vertcat(x1_e, y1_e, theta_e)

    # Converting to AcadosModel
    model = AcadosModel()
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.p = x_goal

    return model

class MPC:
    def __init__(self, model_acados, N=10, t_horizon=5.0, x_goal= np.array([0., 0., 0.])):
        self.N = N # prediction horizon discrete steps
        self.model = model_acados
        self.t_horizon = t_horizon
        self.x_goal = x_goal

        self.solver = self.init_solver()

    def init_solver(self):

        model = self.model
        t_horizon = self.t_horizon
        N = self.N

        # Dimensions: TODO: set the dimensions in the model
        nx = model.x.size()[0] # number of states
        nu = model.u.size()[0] # number of control inputs
        # ny = nx # number of ? nx+nu

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = N 
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        # ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        # Add parameters to the solver
        x_goal = self.x_goal #.reshape(-1, 1)
        ocp.parameter_values = x_goal

        # Initialize cost function. info can be found here: https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf
        # for quadratic cost example see: https://github.com/TUM-AAS/ml-casadi/blob/44ec47f0d14aa8306d0f5dc786babddfbc8964f9/examples/mpc_mlp_naive_example.py#L137

        # the 'EXTERNAL' cost type can be used to define general cost terms
        # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.

        Q_mat = 2*np.diag([0, 0, 0])
        R_mat = 2*np.diag([1, 1])
        Qe_mat = 2*np.diag([100, 100, 100])
        self.Qe_mat = Qe_mat
        x_goal = self.x_goal.reshape(-1, 1)

        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost = model.x.T @ Q_mat @ model.x + model.u.T @ R_mat @ model.u
        ocp.model.cost_expr_ext_cost_e = (model.x - model.p).T @ Qe_mat @ (model.x - model.p)

        # Initial reference trajectory (will be overwritten)
        # ocp.cost.yref = np.zeros(1)

        # Initial state (will be overwritten)
        ocp.constraints.x0 = np.zeros(nx)

        # Set constraints
        v_forward_max = 0.8
        v_backwards_max = 0.8
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

        return AcadosOcpSolver(ocp)
    
    def get_action(self, x_current):
        # This ensures that the first x is the current state
        self.solver.set(0, "lbx", x_current)
        self.solver.set(0, "ubx", x_current)
        self.solver.solve()
        ut = self.solver.get(0, "u")
        xt = self.solver.get(1, "x")
        return ut
    
    def get_state_trajectory(self, x_current):
        # This ensures that the first x is the current state
        self.solver.set(0, "lbx", x_current)
        self.solver.set(0, "ubx", x_current)
        self.solver.solve()
        x_l = np.zeros((self.N, self.model.x.size()[0]))
        for i in range(self.N):
            x_l[i,:] = self.solver.get(i, "x")
        return x_l
    
    def get_action_trajectory(self, x_current):
        # This ensures that the first x is the current state
        self.solver.set(0, "lbx", x_current)
        self.solver.set(0, "ubx", x_current)
        self.solver.solve()
        ut = self.solver.get(0, "u")
        xt = self.solver.get(1, "x")
        u_horizon = np.zeros((self.N, self.model.u.size()[0]))
        for i in range(self.N):
            u_horizon[i,:] = self.solver.get(i, "u")
        return ut, u_horizon, xt
    
    def set_goal(self, x_goal): # TODO: make this change a parameter in the model
        for i in range(self.N):
            self.solver.set(i, "p", x_goal)
        self.x_goal = x_goal


class AcadosTrajectoryOptimizer:
    def __init__(self) -> None:
        N = 30 # discretization steps
        t_horizon = 3.
        x_goal = np.array([0., 0., 0.])
        model = unicycle_ode_model()
        self.mpc = MPC(model_acados=model, N=N, t_horizon=t_horizon, x_goal=x_goal)

    def compute_strategy(self, state, goal=None, obstacle=None):
        assert len(state) == 3
        state = np.array(state)
        now = time.time()
        ut, u_horizon, xt = self.mpc.get_action_trajectory(x_current=state)
        elapsed = time.time() - now
        print(f'Iteration time: {1000*elapsed:.2f}ms')
        print(f'Next action: {ut}')
        return self._get_strategy_from_trajectory(u_horizon)

    def _get_strategy_from_trajectory(self, trajectory):
        """
        Extract the control inputs from the optimized trajectory.

        The robot expects longitudinal velocity (v) and turn rate (\omega) as inputs but the
        trajectory has states [px, py, v, \theta] and inputs [a, \omega].
        """
        control_inputs = []
        trajecotry_length = len(trajectory)

        for i in range(0, trajecotry_length):
            velocity = trajectory[i][0]
            turn_rate = trajectory[i][1]
            control_inputs.append([velocity, turn_rate])

        return OpenLoopStrategy(control_inputs)

def simulate_as_fast(run_many_times=100):
    N = 30 # discretization steps
    t_horizon = 3.
    x_goal = np.array([-5., -5., 0.])

    model = unicycle_ode_model()
    mpc = MPC(model_acados=model, N=N, t_horizon=t_horizon, x_goal=x_goal)

    xt = np.array([0., 0., 0.])
    x = np.zeros((model.x.size()[0], run_many_times))
    x[:,0] = xt
    u = np.zeros((model.u.size()[0], run_many_times))

    ts = t_horizon / N
    opt_times = []

    for i in range(run_many_times):
        now = time.time()

        if i == 50:
            x_goal = np.array([0., 0., 0.])
            mpc = MPC(model_acados=model, N=N, t_horizon=t_horizon, x_goal=x_goal)
            mpc.set_goal(x_goal)
            print('Changed goal')
        
        ut, u_horizon, xt = mpc.get_action_trajectory(x_current=xt)
        x[:,i] = xt
        u[:,i] = ut

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
    simulate_as_fast()
