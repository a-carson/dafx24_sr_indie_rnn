import copy
import json
import torch
import numpy as np


def vector_to_tuple(x):
    num_states = x.shape[-1]
    assert ((num_states % 2) == 0)
    return x[..., :num_states // 2], x[..., num_states // 2:]


def tuple_to_vector(t):
    return torch.cat(t, dim=-1)


def lstm_cell_forward(cell_function, x, h):
    hc = cell_function(x, vector_to_tuple(h))
    return tuple_to_vector(hc)


def rnn_cell_forward(cell_function, x, h):
    return cell_function(x, h)


class AudioRNN(torch.nn.Module):

    def __init__(self,  cell_type: str,
                        hidden_size: int,
                        in_channels: int = 1,
                        out_channels: int = 1,
                        num_layers: int = 1,
                        residual_connection: bool = True):
        super().__init__()
        if cell_type.lower() == 'gru':
            self.rec = torch.nn.GRU(input_size=in_channels, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        elif cell_type.lower() == 'lstm':
            self.rec = torch.nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        else:
            self.rec = torch.nn.RNN(hidden_size=hidden_size, input_size=in_channels, batch_first=True, num_layers=num_layers)

        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=out_channels)
        self.residual = residual_connection
        self.state = None

    def forward(self, x):
        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(2)

        states, last_state = self.rec(x, self.state)
        out = self.linear(states)
        if self.residual:
            out += x[..., 0].unsqueeze(-1)

        self.state = last_state

        if ndim == 2:
            out = out.squeeze(2)

        return out, states

    def reset_state(self):
        self.state = None

    def detach_state(self):
        if type(self.state) is tuple:
            hidden = self.state[0].detach()
            cell = self.state[1].detach()
            if len(self.state) > 2:
                other = self.state[2].detach()
                self.states = (hidden, cell, other)
            else:
                self.state = (hidden, cell)
        else:
            self.state = self.state.detach()


class STN_RNN(torch.nn.Module):
    def __init__(self, cell: torch.nn.RNNCellBase,
                 os_factor=1.0):
        super().__init__()
        self.os_factor = os_factor
        self.cell = cell
        self.hidden_size = self.cell.hidden_size
        if type(self.cell) == torch.nn.LSTMCell:
            self.cell_forward = lstm_cell_forward
        else:
            self.cell_forward = rnn_cell_forward

    def forward(self, x, h=None):

        batch_size = x.shape[0]
        num_samples = x.shape[1]
        k = 1/self.os_factor

        if type(self.cell) == torch.nn.LSTMCell:
            state_size = 2 * self.hidden_size
        else:
            state_size = self.hidden_size

        if h is None:
            h = torch.zeros(batch_size, state_size, device=x.device)
        states = torch.zeros(batch_size, num_samples, state_size, device=x.device)


        for i in range(num_samples):

            xi = x[:, i, :]
            nl = self.nonlinearity(xi, h)
            h = h + k * nl

            states[:, i, :] = h


        return states[..., :self.hidden_size], h

    def nonlinearity(self, x, h):
        h_next = self.cell_forward(self.cell.forward, x, h)
        return h_next - h


class LIDL_RNN(torch.nn.Module):
    def __init__(self, cell: torch.nn.RNNCellBase,
                 os_factor=1.0):
        super().__init__()
        self.os_factor = os_factor
        self.cell = cell
        self.hidden_size = self.cell.hidden_size

        if type(self.cell) == torch.nn.LSTMCell:
            self.cell_forward = lstm_cell_forward
        else:
            self.cell_forward = rnn_cell_forward

    def forward(self, x, h=None):

        batch_size = x.shape[0]
        num_samples = x.shape[1]

        # lin. interp setup
        delay_near = int(np.floor(self.os_factor))
        delay_far = int(np.ceil(self.os_factor))
        alpha = self.os_factor - delay_near

        if type(self.cell) == torch.nn.LSTMCell:
            state_size = 2 * self.hidden_size
        else:
            state_size = self.hidden_size

        if h is None:
            states = torch.zeros(batch_size, num_samples, state_size, device=x.device)
        else:
            states = tuple_to_vector(h)

        for i in range(x.shape[1]):
            # lin. interp
            h_read = (1 - alpha) * states[:, i - delay_near, :] + alpha * states[:, i - delay_far, :]
            xi = x[:, i, :]
            h = self.cell_forward(self.cell.forward, xi, h_read)
            states[:, i, :] = h

        return states[..., :self.hidden_size], vector_to_tuple(states)


class CIDL_RNN(torch.nn.Module):
    def __init__(self, cell: torch.nn.RNNCellBase,
                 os_factor=1.0):
        super().__init__()
        self.os_factor = os_factor
        self.cell = cell
        self.hidden_size = self.cell.hidden_size

        if type(self.cell) == torch.nn.LSTMCell:
            self.cell_forward = lstm_cell_forward
        else:
            self.cell_forward = rnn_cell_forward

    def forward(self, x, h=None):

        batch_size = x.shape[0]
        num_samples = x.shape[1]

        # lagrange interp setup
        order = 3
        if self.os_factor >= 2:
            delta = self.os_factor - np.floor(self.os_factor) + 1
            epsilon = int(np.floor(self.os_factor)) - 1
        else:
            delta = self.os_factor - 1
            epsilon = int(np.floor(self.os_factor))


        kernel = torch.zeros(order+1)
        kernel[0] = -(delta - 1) * (delta - 2) * (delta - 3) / 6
        kernel[1] = delta * (delta - 2) * (delta - 3) / 2
        kernel[2] = -delta * (delta - 1) * (delta - 3) / 2
        kernel[3] = delta * (delta - 1) * (delta - 2) / 6


        if type(self.cell) == torch.nn.LSTMCell:
            state_size = 2 * self.hidden_size
        else:
            state_size = self.hidden_size

        if h is None:
            states = torch.zeros(batch_size, num_samples, state_size, device=x.device)
        else:
            states = tuple_to_vector(h)

        for i in range(x.shape[1]):
            # lin. interp
            h_read = kernel[0] * states[:, i - epsilon, :] + \
                     kernel[1] * states[:, i - epsilon - 1, :] + \
                     kernel[2] * states[:, i - epsilon - 2, :] + \
                     kernel[3] * states[:, i - epsilon - 3, :]

            xi = x[:, i, :]
            h = self.cell_forward(self.cell.forward, xi, h_read)
            states[:, i, :] = h

        return states[..., :self.hidden_size], vector_to_tuple(states)


class APDL_RNN(torch.nn.Module):
    def __init__(self, cell: torch.nn.RNNCellBase,
                 os_factor=1.0):
        super().__init__()
        self.os_factor = os_factor
        self.cell = cell
        self.hidden_size = self.cell.hidden_size

        if type(self.cell) == torch.nn.LSTMCell:
            self.cell_forward = lstm_cell_forward
        else:
            self.cell_forward = rnn_cell_forward

    def forward(self, x, h=None):

        batch_size = x.shape[0]
        num_samples = x.shape[1]

        # lin. interp setup
        delay_near = int(np.floor(self.os_factor))
        delay_far = int(np.floor(self.os_factor)) + 1
        alpha = self.os_factor - delay_near
        allpass_coeff = (1 - alpha) / (1 + alpha)

        if type(self.cell) == torch.nn.LSTMCell:
            state_size = 2 * self.hidden_size
        else:
            state_size = self.hidden_size

        states = torch.zeros(batch_size, num_samples, state_size, device=x.device)
        ap_state = torch.zeros(batch_size, state_size, device=x.device)

        if h is not None:
            states[..., :self.hidden_size] = h[0]
            states[..., self.hidden_size:] = h[1]
            ap_state = h[2]

        for i in range(x.shape[1]):
            ap_state = (1 - alpha) / (1 + alpha) * (states[:, i - delay_near, :] - ap_state) + states[:, i - delay_far, :]
            xi = x[:, i, :]
            h_write = self.cell_forward(self.cell.forward, xi, ap_state)
            states[:, i, :] = h_write

        h = (states[..., :self.hidden_size], states[..., self.hidden_size:], ap_state)
        return states[..., :self.hidden_size], h


def get_AudioRNN_from_json(filename: str):
    with open(filename, 'r') as f:
        json_data = json.load(f)

    model_data = json_data["model_data"]

    model = AudioRNN(cell_type=model_data["unit_type"],
                    in_channels=model_data["input_size"],
                    out_channels=model_data["output_size"],
                    hidden_size=model_data["hidden_size"],
                    residual_connection=bool(model_data["skip"]))

    state_dict = {}
    for key, value in json_data["state_dict"].items():
        state_dict[key.replace("lin", "linear")] = torch.tensor(value)

    model.load_state_dict(state_dict)
    return model


def get_SRIndieRNN(base_model: AudioRNN, method: str):
    model = copy.deepcopy(base_model)

    base_rnn = base_model.rec
    if type(base_model.rec) == torch.nn.LSTM:
        cell = torch.nn.LSTMCell(hidden_size=base_rnn.hidden_size,
                                 input_size=base_rnn.input_size,
                                 bias=base_rnn.bias)
    elif type(base_model.rec) == torch.nn.GRU:
        cell = torch.nn.GRUCell(hidden_size=base_rnn.hidden_size,
                                input_size=base_rnn.input_size,
                                bias=base_rnn.bias)
    else:
        cell = torch.nn.RNNCell(hidden_size=base_rnn.hidden_size,
                                input_size=base_rnn.input_size,
                                bias=base_rnn.bias)
    cell.weight_hh = base_rnn.weight_hh_l0
    cell.weight_ih = base_rnn.weight_ih_l0
    cell.bias_hh = base_rnn.bias_hh_l0
    cell.bias_ih = base_rnn.bias_ih_l0


    if method == 'lidl':
        model.rec = LIDL_RNN(cell=cell)
    elif method == 'apdl':
        model.rec = APDL_RNN(cell=cell)
    elif method == 'stn':
        model.rec = STN_RNN(cell=cell)
    elif method == 'cidl':
        model.rec = CIDL_RNN(cell=cell)

    return model


