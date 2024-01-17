from acados_template import AcadosSimSolver, AcadosOcpSolver, AcadosSim, AcadosOcp, AcadosModel
from casadi import SX, vertcat, sin, cos
import numpy as np

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
    def __init__(self, model_acados, N):
        self.N = N # prediction horizon
        self.model = model_acados

    @property
    def simulator(self):
        return AcadosSimSolver(self.sim())

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())

    def sim(self):
        model = self.model

        t_horizon = 1. # simulation time
        N = self.N

        # Get acados model
        model_ac = model
        model_ac.p = model.p

        # Dimensions: TODO: set the dimensions in the model
        nx = 3 # number of states
        nu = 2 # number of control inputs
        ny = nx # number of outputs

        # Create OCP object to formulate the optimization
        sim = AcadosSim()
        sim.model = model_ac
        sim.dims.N = N
        sim.dims.nx = nx
        sim.dims.nu = nu
        sim.dims.ny = ny
        sim.solver_options.tf = t_horizon # final time of integration

        # Solver options
        sim.solver_options.Tsim = 1./ 10. 
        sim.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        sim.solver_options.hessian_approx = 'GAUSS_NEWTON'
        sim.solver_options.integrator_type = 'ERK'
        # ocp.solver_options.print_level = 0
        sim.solver_options.nlp_solver_type = 'SQP_RTI'

        return sim

    def ocp(self):
        model = self.model

        t_horizon = 1.
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)
        model_ac.p = model.p

        # Dimensions
        nx = 2
        nu = 1
        ny = 1

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        # Initialize cost function
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        ocp.cost.W = np.array([[1.]])

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[0, 0] = 1.
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vz = np.array([[]])
        ocp.cost.Vx_e = np.zeros((ny, nx))
        ocp.cost.W_e = np.array([[0.]])
        ocp.cost.yref_e = np.array([0.])

        # Initial reference trajectory (will be overwritten)
        ocp.cost.yref = np.zeros(1)

        # Initial state (will be overwritten)
        ocp.constraints.x0 = model.x_start

        # Set constraints
        a_max = 10
        ocp.constraints.lbu = np.array([-a_max])
        ocp.constraints.ubu = np.array([a_max])
        ocp.constraints.idxbu = np.array([0])

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        ocp.parameter_values = model.parameter_values

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

if __name__ == '__main__':
    print('test')
    model = unicycle_ode_model()
    print(model.x)
    print(model.p)