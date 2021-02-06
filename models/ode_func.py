import torch
import torch.nn as nn

# git clone https://github.com/rtqichen/torchdiffeq.git
from torchdiffeq import odeint as odeint


class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, latents,
                 odeint_rtol=1e-4, odeint_atol=1e-5, device=torch.device("cpu")):
        super(DiffeqSolver, self).__init__()
        
        self.ode_method = method
        self.latents = latents
        self.device = device
        self.ode_func = ode_func
        
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
    
    def forward(self, first_point, time_steps_to_predict, backwards=False):
        """
		# Decode the trajectory through ODE Solver
		"""
        # n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        
        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1, 0, 2, 3, 4)  # => [b, t, c, h0, w0]

        
        return pred_y
    
    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, n_traj_samples=1):
        """
		# Decode the trajectory through ODE Solver using samples from the prior

		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
        func = self.ode_func.sample_next_point_from_prior
        
        pred_y = odeint(func, starting_point_enc, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1, 2, 0, 3)
        return pred_y


#####################################################################################################

class ODEFunc(nn.Module):
    def __init__(self, opt, input_dim, latent_dim, ode_func_net, device=torch.device("cpu")):
        """
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for ODE. Analog of a continous latent state
		"""
        super(ODEFunc, self).__init__()
        
        self.input_dim = input_dim
        self.device = device
        self.opt = opt

        self.gradient_net = ode_func_net
    
    def forward(self, t_local, y, backwards=False):
        """
		Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point
		t_local: current time point
		y: value at the current time point
		"""
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad
    
    def get_ode_gradient_nn(self, t_local, y):
        output = self.gradient_net(y)
        return output
    
    def sample_next_point_from_prior(self, t_local, y):
        """
		t_local: current time point
		y: value at the current time point
		"""
        return self.get_ode_gradient_nn(t_local, y)
