import convlstm
import dataprocessor
from dataprocessor import DataProcessor
from convlstm import ConvLSTM
import numpy as np
import pandas as pd
import torch
import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import cv2
from sklearn.model_selection import train_test_split

'''
TODO:
    - Evaluation function
'''

class FixationPredictor(nn.Module):
    def __init__(self, data_directory, hidden_dim=32, kernel_size=(3, 3), layer_depth=1, batch_size=32, learning_rate=0.001, weight_decay=0.0001, max_epochs=50, sequence_to_one=True):
        # Init Pytorch module
        super(FixationPredictor, self).__init__()

        # Set model hyperparameters
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.layer_depth = layer_depth
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.sequence_to_one = sequence_to_one

        # Load dataset (also sets some member variables)
        self.loadData(data_directory)

        # Instantiate a ConvLSTM layer, responsible for encoding spatial and temporal information
        self.conv_lstm = ConvLSTM(input_dim=self.channel_depth, hidden_dim=self.hidden_dim, kernel_size=self.kernel_size, layer_depth=self.layer_depth)
        self.last_hidden_list = None

        # Instantiate an addition convolutional layer, responsbile for decoding spatial and temporal into probabilities
        self.decoder_conv = nn.Conv2d(in_channels=self.hidden_dim, out_channels=1, kernel_size=1)

        # Check for nvidia hardware acceleration availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.to(self.device)

    def forward(self, x, hidden_state=None):
        # x must be (B, T, C, H, W)

        B, T, C, H, W = x.shape

        output, new_hidden_state = self.conv_lstm(x, hidden_state)
        B_out, T_out, C_out, H_out, W_out = output.shape

        if self.sequence_to_one:    # Output one fixation prediction for the whole sequence
            last_step_output = output[:, -1, :, :, :]               # Get the last output in time
            logits = self.decoder_conv(last_step_output)            # Decode into saliency scores
            logits_flat = logits.view(B, -1)                        # flatten tensor per batch
            probabilities = torch.log_softmax(logits_flat, dim=1)   # Use softmax to compute probabilities
            prob_maps = probabilities.view(B, H_out, W_out)         # Re-shape back into 2D
        else:                       # Output one fixation prediction per frame
            output = output.view(B_out * T_out, C_out, H_out, W_out)    # Re-shape, cause Conv2D only supports 4D tensors
            logits = self.decoder_conv(output)                          # Decode into saliency scores
            logits_flat = logits.view(B, T, -1)                         # flatten per batch per time
            probabilities = torch.log_softmax(logits_flat, dim=2)       # Use softmax to compute probabilities
            prob_maps = probabilities.view(B, T, H_out, W_out)          # Reshape back into frames

        return prob_maps, new_hidden_state

    def train(self):

        # Define optimizer and err functions
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        size = self.dataset_size

        self.train() # probably not necessary, but good practice

        for video_id, loader in self.train_loaders_dict.items():

            hidden_state = None # want to carry the hidden state across samples, but not across videos

            # Train by number of epochs
            for epoch in range(self.max_epochs):
                if (epoch+1) % 5 == 0:
                    print(f"Training [Progress: {100 * ((epoch+1) / max_epochs):<2.0f}%]", end="")

                # Train in batches
                for X_batch, y_batch in loader:
                    optimizer.zero_grad() # set gradients to zero

                    # Move data to device
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    # Forward pass
                    pred, hidden_state = self(X_batch, hidden_state)
                    hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

                    # Calculate the error
                    error = err_func(pred, y)

                    # Backprop and update weights
                    error.backward()
                    optimizer.step()

        return 0

    def loadData(self, data_dir):
        # Load data from dataset
        processor = DataProcessor(data_dir)
        feature_dict, target_dict = processor.getData()

        # Get video ids and split for training and testing
        video_ids = np.unique(feature_dict.keys())
        train_ids, test_ids = train_test_split(video_ids, test_size=0.2, random_state=42)

        # Dicts to store data loaders
        self.train_loaders_dict = {}
        self.test_loaders_dict = {}

        def init_loader(video_ids, loader_dict):

            for video_id in video_ids:

                X_video = feature_dict[video_id]
                y_video = target_dict[video_id]

                X_samples = dataprocessor.chunkVideo(X_video)

                tensor_x = torch.from_numpy(X_samples).float()
                tensor_y = torch.from_numpy(y_video).float()
                dataset = TensorDataset(tensor_x, tensor_y)

                loader_dict[vid_id] = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        init_loader(train_ids, self.train_loaders_dict)
        init_loader(test_ids, self.test_loaders_dict)

        return 0

    def writeVideo(self, video_tensor, output_filename, fps=30):
        video_tensor = video_tensor.squeeze(0)
        video_tensor = video_tensor.cpu().detach()

        numpy_video = (video_tensor * 255).numpy().astype(np.uint8)
        num_frames, height, width = numpy_video.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (width, height)

        out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size, isColor=False)

        if not out.isOpened():
            print(f"Error: Could not write video")
            return

        for i in range(num_frames):
            frame = numpy_video[i, :, :]
            out.write(frame)

        out.release()
        print("Video saved to {output_filename}")

