import os

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from model import ResidualMultiTaskEncoderDecoder as IAEmu
from pyro.infer import MCMC, NUTS

from get_predictions import Predict
sample = "2"

sample1 = [0.81, 0.35, 12.54, 0.26, 12.68, 13.48, 1.0]  ##fixed HOD for sample 1
sample2 = [0.71, 0.14, 11.93, 0.26, 12.05, 12.85, 1.0]  ##fixed HOD for sample 2
sample3 = [0.54, 0.05, 11.61, 0.26, 11.8, 12.6, 1.0]  ##fixed HOD for sample 3

HOD_map = {"1": sample1[2:], "2": sample2[2:], "3": sample3[2:]}
sample_map = {"1": sample1, "2": sample2, "3": sample3}

xi_obs = np.load(f"../../Illustris/measurements/xi_sample{sample}.npy")
omega_obs = np.load(f"../../Illustris/measurements/omega_sample{sample}.npy")
eta_obs = np.load(f"../../Illustris/measurements/eta_sample{sample}.npy")
xi_cov = np.load(f"../../Illustris/measurements/xi_sample{sample}_cov.npy")
omega_cov = np.load(f"../../Illustris/measurements/omega_sample{sample}_cov.npy")
eta_cov = np.load(f"../../Illustris/measurements/eta_sample{sample}_cov.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize your PyTorch model and move it to the GPU
model = Predict(
    IAEmu(),
    "../iaemu.pt",
    "/Users/snehpandya/Projects/IAEmu/src/iaemu_predict/data/x_train_10x.npy",
    "/Users/snehpandya/Projects/IAEmu/src/iaemu_predict/data/y_train_10x.npy",
)


def forward_model(mu_cen: torch.Tensor, mu_sat: torch.Tensor):
    """_summary_

    Args:
        mu_cen (float): central alignment strength
        mu_sat (float): satellite alignment strength

    Returns:
        position-position (xi) correlation function,
        position-orientation (omega) correlation function,
        orientation-orientation (eta) correlation function,
        xi aleatoric uncertainty,
        omega aleatoric uncertainty,
        eta aleatoric uncertainty
    """
    x = torch.cat(
        [
            mu_cen.unsqueeze(0),
            mu_sat.unsqueeze(0),
            torch.tensor(sample_map[sample][2]).unsqueeze(0),
            torch.tensor(sample_map[sample][3]).unsqueeze(0),
            torch.tensor(sample_map[sample][4]).unsqueeze(0),
            torch.tensor(sample_map[sample][5]).unsqueeze(0),
            torch.tensor(sample_map[sample][6]).unsqueeze(0),
        ],
        dim=0,
    ).to(device)

    preds, aleos, _ = model.predict(x=x, predict="all", return_grad=True)
    xi, omega, eta = preds
    xi_aleo, omega_aleo, eta_aleo = aleos
    return xi, omega, eta, xi_aleo, omega_aleo, eta_aleo


# Define the model function with observational data and covariances
def pyro_model(omega_obs: torch.Tensor, omega_cov: torch.Tensor):
    """_summary_

    Args:
        omega_obs (array): orientation-orientation correlation function from TNG300
        omega_cov (matrix): covariance matrix of omega_obs
    """
    fixed_HOD_params = torch.tensor(HOD_map[sample], dtype=torch.float32, device=device)

    # Sample mu_cen and mu_sat as PyTorch tensors and ensure consistent dtype
    mu_cen = pyro.sample("mu_cen", dist.Uniform(-1, 1)).type_as(fixed_HOD_params)
    mu_sat = pyro.sample("mu_sat", dist.Uniform(-1, 1)).type_as(fixed_HOD_params)

    _, omega_pred, _, _, _, _ = forward_model(mu_cen, mu_sat)

    pyro.sample(
        "omega",
        dist.MultivariateNormal(omega_pred, covariance_matrix=omega_cov),
        obs=omega_obs,
    )


if __name__ == "__main__":
    omega_obs = torch.tensor(omega_obs, dtype=torch.float32).to(device)
    omega_cov = torch.tensor(omega_cov, dtype=torch.float32).to(
        device
    ) + 1e-6 * torch.eye(omega_cov.shape[0]).to(device)

    # Run NUTS sampling with Pyro, passing the observational data and covariances
    nuts_kernel = NUTS(pyro_model, step_size=0.005, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_samples=4000, warmup_steps=2000)
    mcmc.run(omega_obs=omega_obs, omega_cov=omega_cov)
    samples = mcmc.get_samples()

    # Access samples
    mu_cen_samples = samples["mu_cen"]
    mu_sat_samples = samples["mu_sat"]

    save_dir = "../../Illustris/hmc_results_4_3_25/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(f"{save_dir}mu_cen_sample_{sample}.npy", mu_cen_samples)
    np.save(f"{save_dir}mu_sat_sample_{sample}.npy", mu_sat_samples)
    
    diagnostics = mcmc.diagnostics()
    ess_mu_cen = diagnostics["mu_cen"]["n_eff"]
    ess_mu_sat = diagnostics["mu_sat"]["n_eff"]
    
    print(f"Effective Sample Size for mu_cen: {ess_mu_cen}")
    print(f"Effective Sample Size for mu_sat: {ess_mu_sat}")
