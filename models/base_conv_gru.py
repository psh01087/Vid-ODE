import torch
import torch.nn as nn

import sys

sys.path.append('../')
sys.path.append('./')

import utils


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        :param input_size: (int, int) / Height and width of input tensor as (height, width).
        :param input_dim: int / Number of channels of input tensor.
        :param hidden_dim: int / Number of channels of hidden state.
        :param kernel_size: (int, int) / Size of the convolutional kernel.
        :param bias: bool / Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor / Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype
        
        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)
        
        self.conv_can = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)
    
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype)
    
    def forward(self, input_tensor, h_cur, mask=None):
        """
        :param self:
        :param input_tensor: (b, c, h, w) / input is actually the target_model
        :param h_cur: (b, c_hidden, h, w) / current hidden and cell states respectively
        :return: h_next, next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)
        
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)
        
        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)
        
        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        
        mask = mask.view(-1, 1, 1, 1).expand_as(h_cur)
        h_next = mask * h_next + (1 - mask) * h_cur
        
        return h_next


class Encoder_z0_ODE_ConvGRU(nn.Module):
    
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, dtype, batch_first=False,
                 bias=True, return_all_layers=False, z0_diffeq_solver=None, run_backwards=None):
        
        super(Encoder_z0_ODE_ConvGRU, self).__init__()
        
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.z0_diffeq_solver = z0_diffeq_solver
        self.run_backwards = run_backwards
        
        ##### By product for visualization
        self.by_product = {}
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))
        
        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)
        
        # last conv layer for generating mu, sigma
        self.z0_dim = hidden_dim[0]
        z = hidden_dim[0]
        self.transform_z0 = nn.Sequential(
            nn.Conv2d(z, z, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(z, z * 2, 1, 1, 0), )
    
    def forward(self, input_tensor, time_steps, mask=None, tracker=None):
    
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        assert (input_tensor.size(1) == len(time_steps)), "Sequence length should be same as time_steps"
        
        last_yi, latent_ys = self.run_ode_conv_gru(
            input_tensor=input_tensor,
            mask=mask,
            time_steps=time_steps,
            run_backwards=self.run_backwards,
            tracker=tracker)
        
        trans_last_yi = self.transform_z0(last_yi)
        
        mean_z0, std_z0 = torch.split(trans_last_yi, self.z0_dim, dim=1)
        std_z0 = std_z0.abs()
        
        return mean_z0, std_z0
    
    def run_ode_conv_gru(self, input_tensor, mask, time_steps, run_backwards=True, tracker=None):
        
        b, t, c, h, w = input_tensor.size()
        
        device = utils.get_device(input_tensor)
        
        # Set initial inputs
        prev_input_tensor = torch.zeros((b, c, h, w)).to(device)
        
        # Time configuration
        # Run ODE backwards and combine the y(t) estimates using gating
        prev_t, t_i = time_steps[-1] + 0.01, time_steps[-1]
        latent_ys = []
        
        time_points_iter = range(0, time_steps.size(-1))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)
        
        for idx, i in enumerate(time_points_iter):
    
            inc = self.z0_diffeq_solver.ode_func(prev_t, prev_input_tensor) * (t_i - prev_t)
            assert (not torch.isnan(inc).any())
            tracker.write_info(key=f"inc{idx}", value=inc.clone().cpu())
            
            ode_sol = prev_input_tensor + inc
            tracker.write_info(key=f"prev_input_tensor{idx}", value=prev_input_tensor.clone().cpu())
            tracker.write_info(key=f"ode_sol{idx}", value=ode_sol.clone().cpu())
            ode_sol = torch.stack((prev_input_tensor, ode_sol), dim=1)  # [1, b, 2, c, h, w] => [b, 2, c, h, w]
            assert (not torch.isnan(ode_sol).any())
            
            if torch.mean(ode_sol[:, 0, :] - prev_input_tensor) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :] - prev_input_tensor))
                exit()
            
            yi_ode = ode_sol[:, -1, :]
            xi = input_tensor[:, i, :]
            
            # only 1 now
            yi = self.cell_list[0](input_tensor=xi,
                                   h_cur=yi_ode,
                                   mask=mask[:, i])
            
            # return to iteration
            prev_input_tensor = yi
            prev_t, t_i = time_steps[i], time_steps[i - 1]
            latent_ys.append(yi)
        
        latent_ys = torch.stack(latent_ys, 1)
        
        return yi, latent_ys
    
    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states
    
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


def get_norm_layer(ch):
    norm_layer = nn.BatchNorm2d(ch)
    return norm_layer


class Encoder(nn.Module):
    
    def __init__(self, input_dim=3, ch=64, n_downs=2):
        super(Encoder, self).__init__()
        
        model = []
        model += [nn.Conv2d(input_dim, ch, 3, 1, 1)]
        model += [get_norm_layer(ch)]
        model += [nn.ReLU()]
        
        for _ in range(n_downs):
            model += [nn.Conv2d(ch, ch * 2, 4, 2, 1)]
            model += [get_norm_layer(ch * 2)]
            model += [nn.ReLU()]
            ch *= 2
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        out = self.model(x)
        return out


class Decoder(nn.Module):
    
    def __init__(self, input_dim=256, output_dim=3, n_ups=2):
        super(Decoder, self).__init__()
        
        model = []
        
        ch = input_dim
        for i in range(n_ups):
            model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
            model += [nn.Conv2d(ch, ch // 2, 3, 1, 1)]
            model += [get_norm_layer(ch // 2)]
            model += [nn.ReLU()]
            ch = ch // 2
        
        model += [nn.Conv2d(ch, output_dim, 3, 1, 1)]
        # model += [nn.Tanh()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        out = self.model(x)
        return out