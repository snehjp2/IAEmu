import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dataset import IAEmu_Dataset
from model import ResidualMultiTaskEncoderDecoder as IAEmu
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

sns.set()

r_bins = [
    0.11441248,
    0.14739182,
    0.18987745,
    0.24460954,
    0.31511813,
    0.40595079,
    0.52296592,
    0.67371062,
    0.8679074,
    1.11808132,
    1.44036775,
    1.85555311,
    2.39041547,
    3.07945165,
    3.96710221,
    5.11061765,
    6.58375089,
    8.48151414,
    10.92630678,
    14.07580982,
]

timestr = time.strftime("%Y%m%d-%H%M%S")


def set_all_seeds(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(num)


class Analysis:
    """
    Class to analyze the results of the IAEmu model on the test set.

    Inputs:
        - args: argparse object containing the following arguments:
            - model_path: path to the model
            - model_name: name of the model
            - x_test: path to the x_test data
            - y_test: path to the y_test data
            - x_train: path to the x_train data
            - y_train: path to the y_train data
            - mc_passes: number of Monte Carlo passes
            - scale: whether to scale the outputs back to original domain
            - true_stds: path to the true standard deviations
            - keeplog: whether to keep the log of the position-position outputs


    Outputs:
        - saves the model predictions, groudn truth data, input data, and metrics to a directory.
    """

    def __init__(self, args):
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self.args.model_path
        self.model_name = self.args.model_name
        self.model = IAEmu()
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()
        self.model.to(self.device)
        self.scale = self.args.scale
        self.true_stds = np.load(self.args.true_stds)
        self.true_std1, self.true_std2, self.true_std3 = (
            self.true_stds[:, 0],
            self.true_stds[:, 1],
            self.true_stds[:, 2],
        )

        self.train_dataset = IAEmu_Dataset(
            np.load(self.args.x_train), np.load(self.args.y_train)
        )
        self.test_dataset = IAEmu_Dataset(
            np.load(self.args.x_test), np.load(self.args.y_test)
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=128, shuffle=False
        )

        self.mc_passes = self.args.mc_passes
        self.keeplog = self.args.keeplog

        self.save_dir = f"{os.path.dirname(self.model_path)}/test_results_{timestr}"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.get_model_predictions()

        if self.mc_passes > 0:
            self.monte_carlo_dropout(self.mc_passes)

        self.save_results()

        # self.calculate_metrics()

    @torch.no_grad()
    def get_model_predictions(self):
        all_outputs = []
        all_preds = []
        all_inputs = []

        for batch_inputs, batch_outputs in tqdm(self.test_dataloader):
            batch_inputs = batch_inputs.to(self.device)
            batch_outputs = batch_outputs.to(self.device)

            predictions = self.model(batch_inputs)

            all_outputs.append(batch_outputs.detach().cpu().numpy())
            all_preds.append(predictions.detach().cpu().numpy())
            all_inputs.append(batch_inputs.detach().cpu().numpy())

        all_outputs = np.concatenate(all_outputs, axis=0).reshape(-1, 3, 20)
        all_preds = np.concatenate(all_preds, axis=0).reshape(-1, 3, 40)
        all_inputs = np.concatenate(all_inputs, axis=0)

        if self.scale:
            all_outputs = self.test_dataset.clean_outputs
            all_preds = self.train_dataset.inverse_transform(
                torch.tensor(all_preds), keeplog=self.keeplog
            )  # (total_samples, 3, 40)
            if not self.keeplog:
                all_outputs[:, 0, :] = np.exp(all_outputs[:, 0, :])

        else:
            all_outputs = all_outputs
            all_preds = all_preds

        all_preds[:, :, 20:] = np.sqrt(all_preds[:, :, 20:])

        self.all_outputs = all_outputs
        self.all_preds = all_preds
        self.all_inputs = all_inputs

        return self.all_outputs, self.all_preds

    def monte_carlo_dropout(self, num_passes):
        self.model.eval()

        total_samples = len(self.test_dataloader.dataset)
        all_preds = torch.zeros(total_samples, num_passes, 3, 40, device=self.device)

        sample_index = 0

        for batch_inputs, _ in tqdm(self.test_dataloader):
            batch_inputs = batch_inputs.to(self.device)
            batch_size = batch_inputs.size(0)

            batch_preds = torch.zeros(batch_size, num_passes, 3, 40, device=self.device)

            for i in range(num_passes):
                predictions = self.model.forward_with_dropout(batch_inputs)
                batch_preds[:, i, :, :] = predictions

            all_preds[sample_index : sample_index + batch_size, :, :, :] = (
                batch_preds.detach().cpu()
            )

            sample_index += batch_size
            del batch_inputs, batch_preds

        all_mean_preds = torch.mean(all_preds, dim=1)[:, :, :]
        all_var_preds = torch.var(all_preds[:, :, :, :20], dim=1)

        if self.scale:
            scaled_means = self.train_dataset.inverse_transform(
                all_mean_preds, keeplog=self.keeplog
            )
            scaled_vars = self.train_dataset.inverse_mc_variance(
                all_var_preds, keeplog=self.keeplog
            )
            all_mc_stds = np.sqrt(scaled_vars)
        else:
            scaled_means = all_mean_preds.detach().cpu().numpy()
            all_mc_stds = np.sqrt(all_var_preds.detach().cpu().numpy())

        self.all_mc_preds = scaled_means
        self.all_mc_stds = all_mc_stds

        return self.all_mc_preds, self.all_mc_stds

    @torch.no_grad()
    def calculate_metrics(self):
        # Assuming self.all_mc_preds contains mean predictions (samples, 3, 20)
        # and self.all_mc_stds contains standard deviations (samples, 3, 20)
        # Assuming self.all_outputs is the ground truth (samples, 3, 20)

        self.metrics = {"mse": [], "rmse": [], "mape": [], "pearson": []}

        for seq in range(3):  # For each sequence
            # Mean predictions for current sequence
            if self.mc_passes > 0:
                preds_mean = self.all_mc_preds[:, seq, :20]

            else:
                preds_mean = self.all_preds[:, seq, :20]
            # preds_mean = self.all_mc_preds[:, seq, :20]
            # Ground truth for current sequence
            outputs = self.all_outputs[:, seq, :]

            # Compute metrics
            mse = mean_squared_error(outputs, preds_mean, multioutput="raw_values")
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(
                outputs, preds_mean, multioutput="raw_values"
            )

            # Pearson correlation, with p-value filtering
            pearson_corr = [
                pearsonr(preds_mean[i, :], outputs[i, :])[0]
                if pearsonr(preds_mean[i, :], outputs[i, :])[1] < 0.05
                else None
                for i in range(preds_mean.shape[0])
            ]

            # Store metrics
            self.metrics["mse"].append(mse)
            self.metrics["rmse"].append(rmse)
            self.metrics["mape"].append(mape)
            self.metrics["pearson"].append(pearson_corr)

        # Convert lists to numpy arrays for easier manipulation and analysis later
        for key in self.metrics.keys():
            self.metrics[key] = np.array(self.metrics[key])

        return self.metrics

    def plot_metrics(self):
        metrics = self.metrics
        seq_names = [
            "Position-Position",
            "Position-Orientation",
            "Orientation-Orientation",
        ]
        metric_names = ["mse", "rmse", "mape"]

        bin_centers = np.array(r_bins)  # Ensure r_bins is an array for plotting

        # Iterate over each metric to plot
        for metric_name in metric_names:
            fig, ax = plt.subplots(
                figsize=(12, 6)
            )  # Create a new figure for each metric

            # Plot each sequence within the same metric figure
            for seq in range(3):
                ax.plot(
                    bin_centers,
                    metrics[metric_name][seq],
                    label=f"{seq_names[seq]}",
                    marker="o",
                )

            ax.set_title(f"{metric_name.upper()} by Bin")
            ax.set_xlabel("Bin Center")
            ax.set_ylabel(metric_name.upper())
            ax.legend()
            ax.grid(True)
            ax.set_xscale("log")  # Log scale for bin centers if appropriate

            plt.tight_layout()
            plt.savefig(
                f"{self.save_dir}/{metric_name}_metrics.png"
            )  # Save each metric figure
            plt.close()

    @torch.no_grad()
    def save_results(self):
        np.save(f"{self.save_dir}/all_outputs.npy", self.all_outputs)
        np.save(f"{self.save_dir}/all_preds.npy", self.all_preds)

        if self.mc_passes > 0:
            np.save(f"{self.save_dir}/all_mc_preds.npy", self.all_mc_preds)
            np.save(f"{self.save_dir}/all_mc_stds.npy", self.all_mc_stds)

        if self.scale:
            np.save(
                f"{self.save_dir}/all_outputs_scaled.npy", self.test_dataset.outputs
            )
            np.save(
                f"{self.save_dir}/all_preds_scaled.npy",
                self.train_dataset.inverse_transform(
                    self.all_preds, keeplog=self.keeplog
                ),
            )
            np.save(
                f"{self.save_dir}/all_mc_preds_scaled.npy",
                self.train_dataset.inverse_transform(
                    self.all_mc_preds, keeplog=self.keeplog
                ),
            )
            np.save(
                f"{self.save_dir}/all_mc_stds_scaled.npy",
                self.train_dataset.inverse_mc_variance(
                    self.all_mc_stds, keeplog=self.keeplog
                ),
            )

        np.save(f"{self.save_dir}/all_inputs.npy", self.all_inputs)

        # np.save(f"{self.save_dir}/metrics.npy", self.metrics)

        print("Results saved")


if __name__ == "__main__":

    def none_or_str(value):
        if value == "None":
            return None
        return value

    def bool_or_str(value):
        if value == "True":
            return True
        elif value == "False":
            return False
        return value

    parser = argparse.ArgumentParser(description="IA Analysis")
    parser.add_argument(
        "--model_path", type=str, default="results", help="Path to model"
    )
    parser.add_argument("--model_name", type=str, default="IA", help="Model name")
    parser.add_argument("--x_test", type=str, default="None", help="Path to x_test")
    parser.add_argument("--y_test", type=str, default="None", help="Path to y_test")
    parser.add_argument("--x_train", type=str, default="None", help="Path to x_train")
    parser.add_argument("--y_train", type=str, default="None", help="Path to y_train")
    parser.add_argument(
        "--mc_passes", type=int, default=2, help="Number of Monte Carlo passes"
    )
    parser.add_argument(
        "--scale", type=bool_or_str, default="False", help="Scale outputs"
    )
    args = parser.parse_args()

    test = Analysis(args)

    test.save_results()
    test.plot_metrics()
