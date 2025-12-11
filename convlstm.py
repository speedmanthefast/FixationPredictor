# this repo used as a guide: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
import torch
from torch import nn

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

        # Down sampler
        self.encoder = nn.Sequential(
            # Downsample 2x
            nn.Conv2d(input_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            # Downsample 2x (Total 4x)
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.cells = nn.ModuleList()
        for i in range(layer_depth):
            # The input to the first cell is now the output of the encoder (hidden_dim)
            # All subsequent cells also communicate with hidden_dim
            self.cells.append(ConvLSTMCell(hidden_dim, hidden_dim, self.kernel_size, self.bias))

        self.decoder = nn.Sequential(
            # Upsample 2x
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            # Upsample 2x
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )


    def forward(self, input_tensor, hidden_state=None):
        # input_tensor shape: (B, T, C, H, W)
        batch_size, time_steps, _, height, width = input_tensor.size()

        # Flatten time into batch to apply 2D Conv
        x_flat = input_tensor.view(batch_size * time_steps, self.input_dim, height, width)

        # Apply Encoder
        encoded_flat = self.encoder(x_flat)

        # Unflatten back to sequence
        # New shape: (B, T, hidden_dim, H/4, W/4)
        _, c_enc, h_enc, w_enc = encoded_flat.size()
        encoded_sequence = encoded_flat.view(batch_size, time_steps, c_enc, h_enc, w_enc)

        if hidden_state is None:
            # Initialize hidden state with the DOWN-SAMPLED size
            hidden_state = self.init_hidden(batch_size=batch_size, image_size=(h_enc, w_enc))

        last_hidden_list = []    # Stores the final states of each layer
        current_layer_input = encoded_sequence

        for layer in range(self.layer_depth):
            h, c = hidden_state[layer]
            layer_outputs = []

            for time in range(time_steps):

                # Random state drop out to prevenet stickyness (didn't work at all)
                # if self.training and torch.rand(1).item() < 0.1:
                #     h = torch.zeros_like(h)
                #     c = torch.zeros_like(c)

                # Process one time step
                step_input = current_layer_input[:, time, :, :, :]
                h, c = self.cells[layer](step_input, [h, c])
                layer_outputs.append(h)

            # Stack outputs to form the sequence for the next layer
            current_layer_input = torch.stack(layer_outputs, dim=1)
            last_hidden_list.append((h, c))

        lstm_output = current_layer_input # (B, T, hidden_dim, H/4, W/4)

        # Flatten time again
        lstm_output_flat = lstm_output.view(batch_size * time_steps, self.hidden_dim, h_enc, w_enc)

        # Apply Decoder
        decoded_flat = self.decoder(lstm_output_flat)

        # Unflatten
        # Final Shape: (B, T, hidden_dim, H, W)
        _, c_dec, h_dec, w_dec = decoded_flat.size()
        output = decoded_flat.view(batch_size, time_steps, c_dec, h_dec, w_dec)

        return output, last_hidden_list

    def init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.layer_depth):
            init_states.append(self.cells[i].init_hidden(batch_size, image_size))
        return init_states
