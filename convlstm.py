# this repo used as a guide: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim                              # input from previous layer
        self.hidden_dim = hidden_dim                            # num channels for hidden state
        self.kernel_size = kernel_size                          # size of convolution kernel, should be a tuple
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # calculate the padding
        self.bias = bias                                        # should include bias? t/f

        # Convolve height and width.
        # The input will be the input tensor + the hidden state tensor
        # Output will be 4 * the hidden dimension size (one for each gate)
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                            out_channels=4 * self.hidden_dim,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            bias=self.bias)

    def forward(self, input_tensor, current_state):
        current_h, current_c = current_state
        combined = torch.cat([input_tensor, current_h], dim=1)

        combined_conv = self.conv(combined)

        # Split the 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i) # Input gate
        f = torch.sigmoid(cc_f) # Forget gate
        o = torch.sigmoid(cc_o) # Output gate
        g = torch.tanh(cc_g)    # Candidate cell state

        # Calculate new cell state
        c_next = f * current_c + i * g
        # Calculate new hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_depth, kernel_size, bias=True):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim                              # input from previous layer
        self.hidden_dim = hidden_dim                            # num channels for hidden state
        self.kernel_size = kernel_size                          # size of convolution kernel, should be a tuple
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # calculate the padding
        self.bias = bias                                        # should include bias? t/f
        self.layer_depth = layer_depth                          # how many convolutions per layer?

        self.cells = []
        for i in range(layer_depth):
            if i == 0:
                cell_input = self.input_dim
            else:
                cell_input = self.hidden_dim

            # Create cell list
            self.cells.append(ConvLSTMCell(cell_input, self.hidden_dim, self.kernel_size, self.bias))

        # register with pytorch
        self.cells = nn.ModuleList(self.cells)

    def forward(self, input_tensor, hidden_state=None):
        batch_size, time_steps, _, height, width = input_tensor.size()

        # initial hidden states (one for each layer)
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size=batch_size, image_size=(height, width))

        all_outputs = []        # Stores the outputs of each layer
        last_hidden_list = []    # Stores the final states of each layer
        current_layer_inputs = input_tensor

        for layer in range(self.layer_depth):
            h, c = hidden_state[layer] # Get the initial hidden states for each layer (which depends on hidden_dim)
            layer_outputs = []          # Stores the hidden state of each point in time

            for time in range(time_steps):
                h, c = self.cells[layer](input_tensor=current_layer_inputs[:, time, :, :, :], current_state=[h, c]) # Generate next hidden and cell state
                layer_outputs.append(h)         # Save h for later

            layer_output_stack = torch.stack(layer_outputs, dim=1)
            current_layer_inputs = layer_output_stack

            all_outputs.append(layer_output_stack)
            last_hidden_list.append((h, c))

        return all_outputs[-1], last_hidden_list

    def init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.layer_depth):
            init_states.append(self.cells[i].init_hidden(batch_size, image_size))
        return init_states
