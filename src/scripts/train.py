import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import IAEmu_Dataset
from model import ResidualMultiTaskEncoderDecoder as IAEmu
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils import (
    beta_nll_loss,
    combined_val_loss_fn,
    create_directory,
    plot_MAPE,
    plot_predictions,
    plot_rmse,
    save_to_text,
    set_all_seeds,
)

##beta_nll_loss from arxiv:2203.09168

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_path: str,
    optimizer: optim.Optimizer,
    epochs: int = 500,
    early_stopping_patience: int = 100,
    report_interval: int = 5,
    warmup_period: int = 100,
    scheduler: optim.lr_scheduler = None,
):
    """
    Code for training IAEmu model.
    Inputs:
        - model: IAEmu model
        - train_loader: DataLoader for training data
        - val_loader: DataLoader for validation data
        - save_path: path to save model and results
        - optimizer: optimizer for training (uses AdamW)
        - epochs: number of epochs to train
        - early_stopping_patience: early stopping patience based on validation loss
        - report_interval: interval for reporting training and validation loss
        - warmup_period: number of epochs for warmup period
        - scheduler: learning rate scheduler


    Outputs:
        - Saves model, training and validation loss curves, and plots of predictions, RMSE, and MAPE

    Run script with parser arguments shown below.
    """

    criterion = beta_nll_loss
    val_criterion1 = nn.GaussianNLLLoss()
    val_criterion2 = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    (
        train_loss_values,
        val_loss_MSE_values,
        val_loss_NLL_values,
        combined_val_loss_values,
    ) = [], [], [], []
    best_val_loss_MSE, best_val_loss_NLL, best_combined_loss = np.inf, np.inf, np.inf
    no_improvement_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
            position=0,
            leave=True,
        )

        for idx, (batch_inputs, batch_outputs) in progress_bar:
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)
            output1, output2, output3 = (
                batch_outputs[:, 0, :],
                batch_outputs[:, 1, :],
                batch_outputs[:, 2, :],
            )
            optimizer.zero_grad()

            predictions = model(batch_inputs)
            pred1, pred2, pred3 = (
                predictions[:, 0, :20],
                predictions[:, 1, :20],
                predictions[:, 2, :20],
            )
            var1, var2, var3 = (
                predictions[:, 0, 20:],
                predictions[:, 1, 20:],
                predictions[:, 2, 20:],
            )

            if epoch < warmup_period:
                loss1 = criterion(pred1, var1, output1, beta=1.0)
                loss2 = criterion(pred2, var2, output2, beta=1.0)
                loss3 = criterion(pred3, var3, output3, beta=1.0)

            else:
                loss1 = criterion(pred1, var1, output1, beta=0.9)
                loss2 = criterion(pred2, var2, output2, beta=0.5)
                loss3 = criterion(pred3, var3, output3, beta=0.5)

            total_loss = loss1 + loss2 + loss3

            if torch.isnan(total_loss):
                print("Nan loss encountered")
                break

            total_loss.backward()

            clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            train_loss += total_loss.item()

        train_loss /= len(train_loader)
        print(
            f"Epoch [{epoch + 1}/{epochs}] - Batch [{idx + 1}/{len(train_loader)}] - Training Loss: {train_loss}"
        )
        train_loss_values.append(train_loss)

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % report_interval == 0:
            model.eval()
            val_loss_NLL, val_loss_MSE = 0, 0
            with torch.no_grad():
                for idx, (batch_inputs, batch_outputs) in enumerate(val_loader):
                    batch_inputs = batch_inputs.to(device)
                    batch_outputs = batch_outputs.to(device)
                    output1, output2, output3 = (
                        batch_outputs[:, 0, :],
                        batch_outputs[:, 1, :],
                        batch_outputs[:, 2, :],
                    )

                    predictions = model(batch_inputs)
                    pred1, pred2, pred3 = (
                        predictions[:, 0, :20],
                        predictions[:, 1, :20],
                        predictions[:, 2, :20],
                    )
                    var1, var2, var3 = (
                        predictions[:, 0, 20:],
                        predictions[:, 1, 20:],
                        predictions[:, 2, 20:],
                    )

                    loss1_NLL = val_criterion1(pred1, output1, var1)
                    loss2_NLL = val_criterion1(pred2, output2, var2)
                    loss3_NLL = val_criterion1(pred3, output3, var3)

                    total_loss_NLL = loss1_NLL + loss2_NLL + loss3_NLL

                    loss1_MSE = val_criterion2(pred1, output1)
                    loss2_MSE = val_criterion2(pred2, output2)
                    loss3_MSE = val_criterion2(pred3, output3)

                    total_loss_MSE = loss1_MSE + loss2_MSE + loss3_MSE

                    val_loss_MSE += total_loss_MSE.item()
                    val_loss_NLL += total_loss_NLL.item()

            val_loss_MSE /= len(val_loader)
            val_loss_NLL /= len(val_loader)
            combined_val_loss = combined_val_loss_fn(
                val_loss_MSE, val_loss_NLL, alpha=0.7
            )

            print(
                f"Epoch [{epoch + 1}/{epochs}] - Batch [{idx + 1}/{len(val_loader)}] - MSE Validation Loss: {val_loss_MSE} - NLL Validation Loss: {val_loss_NLL} - Combined Validation Loss: {combined_val_loss}"
            )
            val_loss_MSE_values.append(val_loss_MSE)
            val_loss_NLL_values.append(val_loss_NLL)
            combined_val_loss_values.append(combined_val_loss)

            if combined_val_loss < best_combined_loss:
                best_combined_loss = combined_val_loss
                no_improvement_count = 0
                if torch.cuda.device_count() > 1:
                    torch.save(
                        model.module.state_dict(),
                        os.path.join(save_path, "best_model.pt"),
                    )
                else:
                    torch.save(
                        model.state_dict(), os.path.join(save_path, "best_model.pt")
                    )
                print(
                    f"New best model saved at epoch {epoch + 1} with combined loss {combined_val_loss:.4f}"
                )
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if torch.cuda.device_count() > 1:
        torch.save(
            model.eval().module.state_dict(), os.path.join(save_path, "final_model.pt")
        )
    else:
        torch.save(model.eval().state_dict(), os.path.join(save_path, "final_model.pt"))
    np.save(os.path.join(save_path, "losses.npy"), np.array(train_loss_values))
    np.save(
        os.path.join(save_path, "val_losses_MSE.npy"), np.array(val_loss_MSE_values)
    )
    np.save(
        os.path.join(save_path, "val_losses_NLL.npy"), np.array(val_loss_NLL_values)
    )
    print(
        f"Best MSE validation loss of {best_val_loss_MSE} and best NLL validaiton loss of {best_val_loss_NLL} at epoch {epoch + 1}"
    )

    plt.figure(figsize=(10, 10))
    plt.plot(train_loss_values)
    plt.title("Training Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f"{save_path}/train_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.plot(val_loss_MSE_values, color="b", label="MSE")
    plt.plot(val_loss_NLL_values, color="r", label="NLL")
    plt.title("Validation Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{save_path}/validation_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.plot(combined_val_loss_values, color="g", label="Combined Validation Loss")
    plt.title("Validation Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{save_path}/combined_validation_loss.png", dpi=300)
    plt.close()


def evaluate(model, PATH, val_loader, model_version="best_model"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    model.load_state_dict(
        torch.load(os.path.join(PATH, f"{model_version}.pt"), map_location=device)
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    model.eval()

    preds1, preds2, preds3 = [], [], []
    outputs1, outputs2, outputs3 = [], [], []
    variances1, variances2, variances3 = [], [], []
    with torch.no_grad():  # Disable gradient computation during evaluation
        for batch_inputs, batch_outputs in val_loader:
            batch_inputs = batch_inputs.to(device)  # torch.Size([128, 7])
            batch_outputs = batch_outputs.to(device)  # torch.Size([128, 3, 20])

            predictions = model(batch_inputs)

            for i in range(batch_outputs.shape[0]):
                output1, output2, output3 = batch_outputs[i]
                outputs1.append(output1.cpu().numpy())
                outputs2.append(output2.cpu().numpy())
                outputs3.append(output3.cpu().numpy())

                pred1, pred2, pred3 = predictions[i]
                preds1.append(pred1[:20].cpu().numpy())
                preds2.append(pred2[:20].cpu().numpy())
                preds3.append(pred3[:20].cpu().numpy())

                variances1.append(pred1[20:].cpu().numpy())
                variances2.append(pred2[20:].cpu().numpy())
                variances3.append(pred3[20:].cpu().numpy())

    preds1 = np.concatenate([p.reshape(1, -1) for p in preds1], axis=0)
    preds2 = np.concatenate([p.reshape(1, -1) for p in preds2], axis=0)
    preds3 = np.concatenate([p.reshape(1, -1) for p in preds3], axis=0)

    outputs1 = np.concatenate([p.reshape(1, -1) for p in outputs1], axis=0)
    outputs2 = np.concatenate([p.reshape(1, -1) for p in outputs2], axis=0)
    outputs3 = np.concatenate([p.reshape(1, -1) for p in outputs3], axis=0)

    variances1 = np.concatenate([p.reshape(1, -1) for p in variances1], axis=0)
    variances2 = np.concatenate([p.reshape(1, -1) for p in variances2], axis=0)
    variances3 = np.concatenate([p.reshape(1, -1) for p in variances3], axis=0)

    std1 = np.sqrt(variances1)
    std2 = np.sqrt(variances2)
    std3 = np.sqrt(variances3)

    pairs = [
        (preds1, std1, outputs1),
        (preds2, std2, outputs2),
        (preds3, std3, outputs3),
    ]
    correlation_numbers = [0, 1, 2]

    for i, (pred, std, output) in enumerate(pairs):
        plot_predictions(
            pred,
            std,
            output,
            PATH,
            model_version,
            correlation_fn_number=correlation_numbers[i],
        )
        plot_rmse(
            pred,
            output,
            PATH,
            model_version,
            correlation_fn_number=correlation_numbers[i],
        )
        plot_MAPE(
            pred,
            output,
            PATH,
            model_version,
            correlation_fn_number=correlation_numbers[i],
        )

    np.save(os.path.join(PATH, f"{model_version}_preds1.npy"), preds1)
    np.save(os.path.join(PATH, f"{model_version}_preds2.npy"), preds2)
    np.save(os.path.join(PATH, f"{model_version}_preds3.npy"), preds3)

    np.save(os.path.join(PATH, f"{model_version}_outputs1.npy"), outputs1)
    np.save(os.path.join(PATH, f"{model_version}_outputs2.npy"), outputs2)
    np.save(os.path.join(PATH, f"{model_version}_outputs3.npy"), outputs3)

    np.save(os.path.join(PATH, f"{model_version}_std1.npy"), std1)
    np.save(os.path.join(PATH, f"{model_version}_std2.npy"), std2)
    np.save(os.path.join(PATH, f"{model_version}_std3.npy"), std3)


def main(args):
    set_all_seeds(42)
    model = IAEmu()
    dataset = IAEmu_Dataset(np.load(args.x_train_path), np.load(args.y_train_path))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    PATH = f"./IASim_results/{model._get_name()}_{timestr}"
    create_directory(PATH)
    save_to_text(f"{PATH}/args.txt", args.__dict__)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[1 * int(args.num_epochs // 3), 2 * int(args.num_epochs // 3)],
        gamma=0.1,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.num_epochs,
        optimizer=optimizer,
        early_stopping_patience=args.early_stopping_patience,
        save_path=PATH,
        report_interval=args.report_interval,
        scheduler=scheduler,
        warmup_period=args.warmup_period,
    )

    model = IAEmu()
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "mps"))

    evaluate(model=model, PATH=PATH, val_loader=val_loader, model_version="best_model")
    evaluate(model=model, PATH=PATH, val_loader=val_loader, model_version="final_model")


if __name__ == "__main__":

    def none_or_str(value):
        if value == "None":
            return None
        return value

    parser = argparse.ArgumentParser()
    parser.add_argument("--x_train_path", type=str, help="path to input data")
    parser.add_argument("--y_train_path", type=str, help="path to output data")
    parser.add_argument("--x_test_path", type=str, help="path to input data")
    parser.add_argument("--y_test_path", type=str, help="path to output data")
    parser.add_argument("--num_epochs", type=int, help="number of epochs", default=500)
    parser.add_argument("--batch_size", type=int, help="batch size", default=128)
    parser.add_argument(
        "--learning_rate", type=float, help="learning rate", default=1e-2
    )
    parser.add_argument(
        "--report_interval", type=int, help="report interval", default=5
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        help="early stopping patience",
        default=100,
    )
    parser.add_argument("--model_name", type=str, help="model name", default="IAEmu")
    parser.add_argument("--transform", type=none_or_str, help="transform", default=None)
    parser.add_argument("--warmup_period", type=int, help="warmup period", default=100)

    args = parser.parse_args()

    main(args)
