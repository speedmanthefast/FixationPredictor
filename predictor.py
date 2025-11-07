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

class FixationPredictor(nn.Module):
    def __init__(self, data_directory, hidden_dim=32, kernel_size=(3, 3), layer_depth=1, batch_size=32, learning_rate=0.001, weight_decay=0.0001):
        # Init Pytorch module
        super(FixationPredictor, self).__init__()

        # Set model hyperparameters
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.layer_depth = layer_depth
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Load dataset (also sets some member variables)
        self.loadData(data_directory)

        # Instantiate a ConvLSTM layer, responsible for encoding spatial and temporal information
        self.conv_lstm = ConvLSTM(input_dim=self.channel_depth, hidden_dim=self.hidden_dim, kernel_size=self.kernel_size, layer_depth=self.layer_depth)

        # Instantiate an addition convolutional layer, responsbile for decoding spatial and temporal into probabilities
        self.decoder_conv = nn.Conv2d(in_channels=self.hidden_dim, out_channels=1, kernel_size=1)

        # Check for nvidia hardware acceleration availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.to(self.device)

    def forward(self, x):
        # x must be (B, T, C, H, W)

        B, T, C, H, W = x.shape

        output = self.conv_lstm(x)
        B_out, T_out, C_out, H_out, W_out = output.shape

        output = output.view(B_out * T_out, C_out, H_out, W_out)
        logits = self.decoder_conv(output)

        logits_flat = logits.view(B, T, -1)
        probabilities = torch.log_softmax(logits_flat, dim=2)

        prob_maps = probabilities.view(B, T, H_out, W_out)

        return prob_maps

    def train_evaluate(self):
        # WRITE EVALUATE FUNCTION LATER

        # Define optimizer and err functions
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        size = self.dataset_size

        self.train() # probably not necessary, but good practice
        max_epochs = 50

        # Train by number of epochs
        for epoch in range(max_epochs):
            if (epoch+1) % 5 == 0:
                print(f"Training [Progress: {100 * ((epoch+1) / max_epochs):<2.0f}%]", end="")

            # Train in batches
            for batch, (X, y) in enumerate(self.train_loader):

                # Prepare data
                X, y = X.to(self.device), y.to(self.device)

                # Calculate error
                pred = self(X)
                error = err_func(pred, y)

                # Backprop and update weights
                error.backward()
                optimizer.step()
                optimizer.zero_grad()

        return 0

    def loadData(self, data_dir):
        processor = DataProcessor(data_dir)
        self.X, self.y = processor.getData()
        self.dataset_size, self.time_steps, self.channel_depth, self.frame_height, self.frame_width = self.X.shape

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

        X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

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


data_directory = "/path/to/data"
model = FixationPredictor(data_directory)

dummy_video_clip = torch.randn(1, model.time_steps, model.channel_depth, model.frame_height, model.frame_width)
print(f"Input shape:    {dummy_video_clip.shape}")

prob_maps = model(dummy_video_clip)
print(f"Output maps shape: {prob_maps.shape}")

model.writeVideo(prob_maps, "prediction.mp4")

# Check that probabilities sum to 1
one_map = prob_maps[0, 0] # First map of first batch
print(f"Sum of probabilities: {one_map.sum().item()}")
