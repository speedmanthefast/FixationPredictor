import torch
import numpy as np
import matplotlib.pyplot as plt

class SaliencyMetrics:
    def __init__(self):

        self.kldcc_history = []

        self.total_cc = 0
        self.total_sim = 0
        self.total_kld = 0
        self.total_nss = 0

        self.batch_counter = 0

    def reset(self):
        self.total_cc = 0
        self.total_sim = 0
        self.total_kld = 0
        self.total_nss = 0
        self.batch_counter = 0

    # averages metrics per num batches
    def summarize(self):
        batch_counter = self.batch_counter

        if batch_counter == 0:
            print("Error: Could not summarize results. Batch count is ZERO!")
            return

        cc = self.total_cc / batch_counter
        sim = self.total_sim / batch_counter
        kld = self.total_kld / batch_counter
        nss = self.total_nss / batch_counter

        print(f"Correlation Coefficient: {cc:.3f}")         # target: 1.0
        print(f"Similarity Metric: {sim:.3f}")              # target: 1.0
        print(f"Kullback-Leibler Divergence: {kld:.3f}")    # target: 0.0
        print(f"Normalized Scanpath Saliency: {nss:.3f}")   # target: >1.0

    def plot_results(self):
        # Calculate a simple moving average (window size 50)
        window_size = 50
        data = torch.tensor(self.kldcc_history)

        # Use a simple convolution to smooth
        smoothed = data.unfold(0, window_size, 1).mean(1).numpy()

        plt.figure(figsize=(10, 6))

        # Plot raw data (as a faint line)
        plt.plot(self.kldcc_history, color='blue', alpha=0.2, label='Raw Batch Loss')

        # Plot moving average (as a solid line cause that more important)
        plt.plot(smoothed, color='red', linewidth=2, label=f'Moving Average ({window_size})')

        # Add titles and labels and stuff
        plt.legend()
        plt.title("KLD-CC-NSS Training Chart", fontsize=16)
        plt.xlabel("Batch", fontsize=12)
        plt.ylabel("KLD-CC-NSS", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Show the plot
        plt.savefig("kldcchistory.png")


    # runs evaluation metrics
    def evaluate(self, preds, y_batch):
        self.batch_counter += 1

        self.total_cc += self.CC(preds, y_batch)
        self.total_sim += self.SIM(preds, y_batch)
        self.total_kld += self.KLD(preds, y_batch)
        self.total_nss += self.NSS(preds, y_batch)

    @staticmethod
    def normalize_map(x):
        # Normalize a map to sum to 1 (as in a probability distributon)
        batch_size = x.size(0)
        view = x.view(batch_size, -1)
        return (x / (view.sum(dim=1, keepdim=True).view(batch_size, 1, 1) + 1e-7))

    @staticmethod
    def normalize_std(x):
        # Normalize to have zero mean and std of 1
        batch_size = x.size(0)
        view = x.view(batch_size, -1)
        std, mean = torch.std_mean(view, dim=1, keepdim=True)
        return (x - mean.view(batch_size, 1, 1)) / (std.view(batch_size, 1, 1) + 1e-7)

    def CC(self, preds, targets):
        # Correlation Coefficient
        # Tells us how well the prediction matches the fixation via element wise products
        # Cares more about the 'where' of prediction rather than the confidence

        # Normalize to 0-mean, 1-std
        p_norm = self.normalize_std(preds)
        t_norm = self.normalize_std(targets)

        # Calculate correlation
        cc = torch.mean(p_norm * t_norm, dim=(1, 2, 3))
        return torch.mean(cc) # Average over batch (even though batch is actually 1 now)

    def SIM(self, preds, targets):
        # Similarity Metric
        # Basically, it measures the intersection of the two distributions
        # Punishes when the distributions are offset

        # Normalize into probability distros
        p_norm = self.normalize_map(preds)
        t_norm = self.normalize_map(targets)

        # Sum of minimums
        sim = torch.sum(torch.min(p_norm, t_norm), dim=(1, 2, 3))
        return torch.mean(sim)

    def KLD(self, preds, targets):
        # Kullback-leibler divergence
        # Measures distance between distributions
        # Heavily punishes NOT predicting saliency where there is saliency

        # Normalize to sum to 1
        p_norm = self.normalize_map(preds)
        t_norm = self.normalize_map(targets)

        kld = torch.sum(t_norm * torch.log(1e-7 + t_norm / (p_norm + 1e-7)), dim=(1, 2, 3))
        return torch.mean(kld)

    def NSS(self, preds, targets):
        # Normalized scanpath saliency
        # Tells us how well the targets fall into high-confidence regions of the prediction
        # This metric cares if the places people looked where high-confidence regions. Doesn't care about false positives
        # This metric will likely be lower than the standard because the target maps are smoothed out

        # Normalize preds to 0-mean, 1-std
        p_norm = self.normalize_std(preds)

        nss = torch.sum(p_norm * targets, dim=(1, 2, 3)) / (torch.sum(targets, dim=(1, 2, 3)) + 1e-7)
        return torch.mean(nss)
