import torch
import pprint
import random
import numpy as np
import os
import matplotlib.pyplot as plt

r_bins = [ 
          0.11441248,  0.14739182,  0.18987745,  0.24460954,  0.31511813,
           0.40595079,  0.52296592,  0.67371062,  0.8679074 ,  1.11808132,
           1.44036775,  1.85555311,  2.39041547,  3.07945165,  3.96710221,
           5.11061765,  6.58375089,  8.48151414, 10.92630678, 14.07580982
           ]

def save_to_text(filename, data):
    """Utility to save data to a pickle file."""
    with open(filename, 'w') as file:
        pprint.pprint(data, stream=file)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_all_seeds(num):
    '''Set all seeds for reproducibility'''
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(num)
        
def beta_nll_loss(pred, variance, target, beta, epsilon=1e-6):
    """Compute beta-NLL loss
    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative
        weighting between data points, where "0" corresponds to
        high weight on low error points and "1" to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    epsilon = torch.tensor(epsilon).to(pred.device)
    stabilized_variance = torch.max(variance, epsilon)
    
    gaussian_nll_loss = 0.5 * ((pred - target) ** 2 / stabilized_variance + torch.log(stabilized_variance))
    
    if beta > 0:
        beta_nll_loss = gaussian_nll_loss * stabilized_variance.detach() ** beta
        
    return beta_nll_loss.sum(axis=-1).mean()

def combined_val_loss_fn(val_loss_MSE, val_loss_NLL, alpha=0.7):
    """
    Calculate a combined loss with a specified weighting.
    The default alpha gives more weight to MSE.
    """
    return alpha * val_loss_MSE + (1 - alpha) * val_loss_NLL


def plot_predictions(predictions, 
                     std, 
                     target, 
                     PATH, 
                     model_version, 
                     correlation_fn_number):
    
    rmses = np.sqrt(np.mean((predictions - target)**2, axis=1))
    mae = np.mean(np.abs(predictions - target), axis=1)
    plt.figure(figsize=(15, 10))
    set_all_seeds(42)
    
    correlation_fn_dict = {0: 'Position-Position', 1: 'Position-Orientation', 2: 'Orientation-Orientation'}
        
    for i in range(25):
        idx = np.random.randint(0, predictions.shape[0])
        plt.subplot(5, 5, i+1)
        plt.errorbar(r_bins, predictions[idx], yerr=std[idx], fmt='o', capsize=5, label='Predicted')
        plt.plot(r_bins, target[idx], label = 'Actual')
        plt.text(0.6, 0.9, f'idx: {idx}', transform=plt.gca().transAxes)
        plt.text(0.6, 0.8, f'RMSE: {rmses[idx]:.3f}', transform=plt.gca().transAxes)
        plt.text(0.6, 0.7, f'MAE: {mae[idx]:.3f}', transform=plt.gca().transAxes)
        plt.xticks([])
        plt.xscale('log')
        if predictions[idx].max() > 10:
            plt.yscale('log')
        
    plt.suptitle(f'Actual vs Predicted, {correlation_fn_dict[correlation_fn_number]}, {model_version}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{PATH}/actual_vs_predicted_{model_version}_{correlation_fn_number}.png', dpi=300)
    plt.close()
    
def plot_rmse(predictions, 
              target, 
              PATH, 
              model_version, 
              correlation_fn_number):
    
    '''Root Mean Squared Error plots.'''
    
    correlation_fn_dict = {0: 'Position-Position', 1: 'Position-Orientation', 2: 'Orientation-Orientation'}

    errors = np.abs(predictions - target) # Shape: [num_samples, seq_len]
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    std = np.std(errors, axis=0) / np.sqrt(errors.shape[0])
    
    plt.plot(rmse)
    plt.fill_between(np.arange(20), rmse - std, rmse + std, alpha=0.2)
    plt.title(f'RMSE, {correlation_fn_dict[correlation_fn_number]}, {model_version}')
    plt.xlabel('$r$ bins')
    plt.xticks(np.arange(20), [str(i) for i in range(1, 21)])
    plt.tight_layout()
    plt.savefig(f'{PATH}/rmse_{model_version}_{correlation_fn_number}.png', dpi=300)
    plt.close()
    
def plot_MAPE(predictions, 
              target, 
              PATH, 
              model_version, 
              correlation_fn_number):
    
    '''Mean Absolute Percentage Error plots.'''
    
    correlation_fn_dict = {0: 'Position-Position', 1: 'Position-Orientation', 2: 'Orientation-Orientation'}
    
    aapd = abs(target - predictions) / (abs(target)) * 100
    column_means = np.mean(aapd, axis=0)
    column_stddev = np.std(aapd, axis=0)
    
    plt.plot(np.arange(20), np.zeros(20), label='0')
    plt.errorbar(np.arange(aapd.shape[1]), column_means, yerr=column_stddev / np.sqrt(aapd.shape[0]), fmt='o', capsize=5, label='Predictions')
    plt.title(f'MAPE, {correlation_fn_dict[correlation_fn_number]}, {model_version}')
    plt.xlabel('$r$ bins')
    plt.ylabel('MAPE', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{PATH}/MAPE_{model_version}_{correlation_fn_number}.png', dpi=300)
    plt.close()
    
    