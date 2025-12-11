from dataprocessor import DataProcessor
from convlstm import ConvLSTM
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
import cv2
from sklearn.model_selection import train_test_split
from videohelpers import chunkVideo, resizeFrame
from evaluation import SaliencyMetrics
import os
import random

class SaliencyLoss(nn.Module):

    def __init__(self, kl=1.0, cc=10.0, nss=0.2):
        super(SaliencyLoss, self).__init__()
        self.kl = kl
        self.cc = cc
        self.nss = nss

    def forward(self, preds, targets):
        # Expect preds and targets to be in batches (B, H, W)

        preds_flattened = preds.view(preds.size(0), -1)
        targets_flattened = targets.view(targets.size(0), -1)

        # KLD
        targets_prob = targets_flattened / (targets_flattened.sum(dim=1, keepdim=True) + 1e-7)
        loss_kl = F.kl_div(preds_flattened, targets_prob, reduction='batchmean')

        # CC
        x = torch.exp(preds_flattened)
        y = targets_flattened
        vx = x - x.mean(dim=1, keepdim=True)
        vy = y - y.mean(dim=1, keepdim=True)
        cc = torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)) + 1e-7)
        loss_cc = cc.mean()

        # NSS
        x_std = x.std(dim=1, keepdim=True)
        x_norm = (x - x.mean(dim=1, keepdim=True)) / (x_std + 1e-7)
        nss_numerator = torch.sum(x_norm * y, dim=1)
        nss_denominator = torch.sum(y, dim=1) + 1e-7
        nss_score = (nss_numerator / nss_denominator).mean()

        return (self.kl * loss_kl) + (self.cc * (1 - loss_cc)) - (self.nss * nss_score)

class FixationPredictor(nn.Module):
    def __init__(self, features, hidden_dim=64, kernel_size=(7, 7), layer_depth=3, batch_size=1, learning_rate=0.001, weight_decay=0.0005, max_epochs=50, accumulation=4): #sequence_to_one=True):
        # Init Pytorch module
        super(FixationPredictor, self).__init__()

        # Set model hyperparameters
        self.batch_size = batch_size        # number of clips to use in a training batch (NOTE: this actually needs to be 1)
        if batch_size > 1:
            print("WARNING: using a batch size of more than 1 will break temporal continuity")
        self.hidden_dim = hidden_dim        # The number of dimensions in the hidden layers
        self.kernel_size = kernel_size      # The size of the convolutional kernel
        self.layer_depth = layer_depth      # The number of convolutional layers per time step
        self.learning_rate = learning_rate  # Affects how much the weights are adjusted after backpropagation
        self.weight_decay = weight_decay    # Decays weights to prevent them from becoming too large
        self.max_epochs = max_epochs        # The number of times to run through the whole dataset during training
        self.accumulation = accumulation    # How many times to accumulate gradients before updating weights (simulates batch training)
        self.channel_depth = len(features)  # The depth of the convolutional kernel. Should correspond to the number of features.
        self.features = features            # the list of chosen features
        self.dropout = nn.Dropout2d(p=0.2)

        self.metrics = SaliencyMetrics()

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

        # Apply Dropout (only during training)
        x_flat = x.view(B * T, C, H, W)
        x_dropped = self.dropout(x_flat)
        x = x_dropped.view(B, T, C, H, W)

        output, new_hidden_state = self.conv_lstm(x, hidden_state)
        B_out, T_out, C_out, H_out, W_out = output.shape

        output = output.view(B_out * T_out, C_out, H_out, W_out)    # Re-shape, cause Conv2D only supports 4D tensors
        logits = self.decoder_conv(output)                          # Decode into saliency scores
        logits_flat = logits.view(B, T, -1)                         # flatten per batch per time
        log_probabilities = torch.log_softmax(logits_flat, dim=2)   # normalize into probability distribution
        log_prob_maps = log_probabilities.view(B, T, H_out, W_out)  # reshape back into 2D

        return log_prob_maps, new_hidden_state

    def fit(self):

        # Define optimizer and err functions
        optimizer = optim.Adam(self.conv_lstm.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        err_func = SaliencyLoss()

        self.train()

        scaler = GradScaler()

        # Train by number of epochs
        for epoch in range(self.max_epochs):
            if epoch % 2 == 0:
                print(f"\rTraining [Progress: {100 * (epoch / self.max_epochs):<2.0f}%]", end="", flush=True)

                video_ids = list(self.train_loaders_dict.keys())
                random.shuffle(video_ids)

            for video_id in video_ids:
                loader = self.train_loaders_dict[video_id]

                hidden_state = None # want to carry the hidden state across samples, but not across videos
                optimizer.zero_grad()

                # Train in batches
                for i, (X_batch, y_batch) in enumerate(loader):

                    # Move data to device
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    # Cast to mixed precision where possible. Greatly reduces GPU memory overhead
                    with autocast('cuda'):
                        pred, hidden_state = self(X_batch, hidden_state)

                        # Detach hidden states to prevent graph from growing infinitely (truncated BPTT)
                        hidden_state = [(h.detach(), c.detach()) for h, c in hidden_state]

                        error = err_func(pred, y_batch) / self.accumulation

                    # Scale the gradients
                    scaler.scale(error).backward()

                    # Every so often, update the weights. We do this to simulate the effect of batch processing (reduce influence of noisy data)
                    # Regular batch processing cannot be used in our architecture because the videos are chunked in to sequences
                    # If we have batch size > 1, the hidden state will not be carred to the logical next sequence in time, but multiple sequences ahead, breaking continuity.
                    if (i + 1) % self.accumulation == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                        # Metrics can run outside autocast (save memory by detaching first)
                        with torch.no_grad():
                            self.metrics.kldcc_history.append(error.item() * self.accumulation)
                            pred_linear = torch.exp(pred.detach())
                            self.metrics.evaluate(pred_linear, y_batch)

        print("\rTraining [Progress: 100%]            ")
        print("TRAINING COMPLETE")
        self.metrics.summarize()
        self.metrics.plot_results()

    def test(self):

        print("BEGINNING TEST PHASE")
        self.metrics.reset()
        self.eval()

        for video_id, loader in self.test_loaders_dict.items():

            for X_batch, y_batch in loader:
                hidden_state = None

                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass
                pred, hidden_state = self(X_batch, hidden_state)
                hidden_state = [(h.detach(), c.detach()) for h, c in hidden_state]

                # Calculate the error
                pred_linear = torch.exp(pred.detach())
                self.metrics.evaluate(pred_linear, y_batch)

        print("TEST PHASE COMPLETE")
        self.metrics.summarize()

    # Runs inference on a single video given its extracted features
    def infer(self, features_dir, output_file="./prediction.mp4"):
        self.eval()

        self.processor = DataProcessor(self.features)

        feature_numpy = self.processor.loadFeatures(features_dir)
        feature_samples = chunkVideo(feature_numpy, self.processor.frames_per_sequence)

        tensor = torch.from_numpy(feature_samples).float()
        num_sequences, t_seq, c, h, w = tensor.shape

        output_frames = []
        hidden_state = None

        # Skip calculating gradients using torch.no_grad()
        with torch.no_grad():
            for sequence in range(num_sequences):

                batch = tensor[sequence].unsqueeze(0).to(self.device)
                output_map, hidden_state = self(batch, hidden_state)
                output_map_linear = torch.exp(output_map)

                prediction = output_map_linear

                output_frames.append(prediction.cpu())

        full_video_tensor = torch.cat(output_frames, dim=1) # concat along time dimension

        print(f"Saving video to {output_file}")
        self.writeVideo(full_video_tensor, output_file, fps=self.processor.fps)

        self.train()

        return full_video_tensor


    def loadData(self, data_dir):
        # Load data from dataset
        self.processor = DataProcessor(self.features)
        self.processor.loadDataset(data_dir)
        feature_dict, target_dict = self.processor.getData()

        # Get video ids and split for training and testing
        video_ids = list(feature_dict.keys())
        random.shuffle(video_ids)

        if len(video_ids) > 0:
            features_loaded = feature_dict[video_ids[0]].shape[1]
            if self.channel_depth != features_loaded:
                raise ValueError(f"DataProcessor loaded {features_loaded}, but {self.channel_depth} were expected")
        else:
            raise ValueError("DataProcessor did not load any data")

        train_ids, test_ids = train_test_split(video_ids, test_size=0.2, random_state=42)

        print(f"Training videos: {train_ids}")
        print(f"Testing videos: {test_ids}")

        # Dicts to store data loaders
        self.train_loaders_dict = {}
        self.test_loaders_dict = {}

        def init_loader(video_ids, loader_dict):

            for video_id in video_ids:

                # Load feature and target numpys
                X_video = feature_dict[video_id]
                y_video = target_dict[video_id]

                # Chunk the video into sequences to match the time scale of the targets
                X_samples = chunkVideo(X_video, self.processor.frames_per_sequence)
                y_samples = chunkVideo(y_video, self.processor.frames_per_sequence)

                # This will ensure the the first dimension of the arrays are equal by dropping samples
                # X_samples, y_samples = self.processor.validate_data(video_id, X_samples, y_samples)
                min_len = min(X_samples.shape[0], y_samples.shape[0])

                # Check for severe mismatch (e.g., if one is double the other, something is wrong)
                if abs(X_samples.shape[0] - y_samples.shape[0]) > 5:
                    print(f"WARNING: Large mismatch in {video_id}. Features: {X_samples.shape[0]}, Targets: {y_samples.shape[0]}")

                # Truncate both to the minimum common length
                print(f"Dropping {X_samples.shape[0]} feature samples and {y_samples.shape[0]} target samples to {min_len}")
                X_samples = X_samples[:min_len]
                y_samples = y_samples[:min_len]

                # Convert numpy arrays to torch tensors / datasets / dataloaders
                tensor_x = torch.from_numpy(X_samples).float()
                tensor_y = torch.from_numpy(y_samples).float()
                dataset = TensorDataset(tensor_x, tensor_y)
                loader_dict[video_id] = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

                print(f"Loaded {video_id} into dataloader | X shape: {tensor_x.shape}, y shape: {tensor_y.shape}")

        init_loader(train_ids, self.train_loaders_dict)
        init_loader(test_ids, self.test_loaders_dict)

        return 0

    def writeVideo(self, video_tensor, output_filename, fps=8):
        video_tensor = video_tensor.squeeze(0)
        video_tensor = video_tensor.cpu().detach()

        numpy_video = video_tensor.numpy()
        num_frames, height, width = numpy_video.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (3840, 1920)

        out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size, isColor=True)

        if not out.isOpened():
            print(f"Error: Could not write video to {output_filename}")
            return

        for i in range(num_frames):
            frame = numpy_video[i, :, :]

            # Resize FIRST to get smooth gradients instead of blocks
            #print(f"Resizing {frame.shape} to {frame_size}")
            frame_resized = resizeFrame(frame, frame_size)

            # Normalize by the max value in the frame
            frame_max = frame_resized.max()
            if frame_max > 1e-7:
                frame_norm = frame_resized / frame_max
            else:
                frame_norm = frame_resized

            # Convert to Heatmap
            heatmap_uint8 = (frame_norm * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            out.write(heatmap_color)

        out.release()
        print(f"Video saved to {output_filename}")

    def save(self, path="model.pth"):
        torch.save(self.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load(self, path="model.pth"):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print(f"Model weights loaded from {path}")
        else:
            print(f"No model found at {path}")

