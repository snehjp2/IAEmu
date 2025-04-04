from typing import List

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class IAEmu_Dataset(Dataset):
    """
    Dataset class for IAEmu model
    """
    def __init__(self, inputs: List, outputs: List):
        super(IAEmu_Dataset, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        self.outputs = torch.tensor(outputs, dtype=torch.float32).to(self.device)

        negative_mask = torch.any(self.outputs[:, 0, :] < 0, dim=1)
        clean_mask = ~negative_mask

        self.clean_outputs = self.outputs[clean_mask]
        self.clean_inputs = self.inputs[clean_mask]
        self.clean_outputs[:, 0, :] = torch.log(self.clean_outputs[:, 0, :])

        self.input_scaler = StandardScaler()
        clean_inputs_cpu = self.clean_inputs.cpu().numpy()
        self.inputs = torch.tensor(
            self.input_scaler.fit_transform(clean_inputs_cpu), dtype=torch.float32
        ).to(self.device)

        self.output_scalers = [StandardScaler() for _ in range(outputs.shape[1])]
        clean_outputs_normalized = []

        for i in range(outputs.shape[1]):
            clean_outputs_cpu = self.clean_outputs[:, i, :].cpu().numpy()
            scaled_output = (
                self.output_scalers[i]
                .fit_transform(clean_outputs_cpu.reshape(-1, 1))
                .reshape(self.clean_outputs[:, i, :].shape)
                .astype(np.float32)
            )
            clean_outputs_normalized.append(
                torch.tensor(scaled_output, dtype=torch.float32).to(self.device)
            )

        self.targets = torch.stack(clean_outputs_normalized, axis=1).to(self.device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        output_sample = self.targets[idx]

        return input_sample, output_sample

    def inverse_transform(self, scaled_outputs, keeplog=False):
        original_outputs = torch.empty_like(scaled_outputs, dtype=torch.float32).to(
            self.device
        )

        for sequence_index in range(3):
            scaler = self.output_scalers[sequence_index]
            scale = torch.tensor(scaler.scale_, dtype=torch.float32).to(self.device)
            mean = torch.tensor(scaler.mean_, dtype=torch.float32).to(self.device)

            if sequence_index == 0:
                log_means_scaled = scaled_outputs[:, sequence_index, :20]
                log_means_original = log_means_scaled * scale + mean
                self.original_means = log_means_original
                self.means1 = log_means_original

                if not keeplog:
                    self.original_means = torch.exp(log_means_original)

                log_variances_scaled = scaled_outputs[:, sequence_index, 20:]
                sigma_squared = scale**2  # Convert sigma^2 to tensor
                adjusted_variances = log_variances_scaled * sigma_squared

                if not keeplog:
                    adjusted_variances = self.original_means * adjusted_variances

            else:
                means_scaled = scaled_outputs[:, sequence_index, :20]
                self.original_means = means_scaled * scale + mean

                variances_scaled = scaled_outputs[:, sequence_index, 20:]
                sigma_squared = scale**2  # Convert sigma^2 to tensor
                adjusted_variances = variances_scaled * sigma_squared

            original_outputs[:, sequence_index, :20] = self.original_means
            original_outputs[:, sequence_index, 20:] = adjusted_variances

        return original_outputs

    def inverse_mc_variance(self, scaled_variances, keeplog=False):
        original_variances = torch.empty_like(scaled_variances, dtype=torch.float32).to(
            self.device
        )

        for sequence_index in range(3):
            scaler = self.output_scalers[sequence_index]
            scale = torch.tensor(
                scaler.scale_, dtype=scaled_variances.dtype, device=self.device
            )

            if sequence_index == 0:
                sigma_squared = scale**2  # Convert sigma^2 to tensor

                log_variances_scaled = scaled_variances[:, sequence_index, :]
                adjusted_variances = log_variances_scaled * sigma_squared

                if not keeplog:
                    adjusted_variances = self.means1 * adjusted_variances

            else:
                sigma_squared = scale**2  # Convert sigma^2 to tensor

                variances_scaled = scaled_variances[:, sequence_index, :]
                adjusted_variances = variances_scaled * sigma_squared

            original_variances[:, sequence_index, :] = adjusted_variances

        return original_variances



if __name__ == "__main__":
    inputs = np.load(
        "/Users/snehpandya/Projects/IAsim/src/data/new_new_new_data/x_test_clean.npy",
        allow_pickle=True,
    )
    outputs = np.load(
        "/Users/snehpandya/Projects/IAsim/src/data/new_new_new_data/y_test_means.npy",
        allow_pickle=True,
    )
    stds = np.load(
        "/Users/snehpandya/Projects/IAsim/src/scripts/scaled_variances.npy",
        allow_pickle=True,
    )

    dataset = IAEmu_Dataset(inputs, outputs)
    print(dataset.inputs[7])
    # print(np.sqrt(dataset.inverse_mc_variance(stds)))
