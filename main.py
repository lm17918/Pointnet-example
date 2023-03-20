import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.datasets import PointCloudData, default_transforms
from src.utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, trainloader, optimizer, pointnetloss):
    model.train()
    for data in tqdm(trainloader, total=len(trainloader), desc="Training", leave=False):
        inputs, labels = data["pointcloud"].to(device).float(), data["category"].to(
            device
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs, m3x3, m64x64 = model(inputs.transpose(1, 2))
            loss = pointnetloss.forward(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item()})


def validation(model, val_loader, decimate):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data in tqdm(val_loader, total=len(val_loader), desc="Eval", leave=False):
            inputs, labels = data["pointcloud"].to(device).float(), data["category"].to(
                device
            )
            outputs, __, __ = model(inputs.transpose(1, 2))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100.0 * correct / total
    wandb.log({"Validation accuracy": val_acc})
    wandb.log({"Weighted validation accuracy": val_acc * (1 - decimate)})
    return val_acc


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.training.seed)
    wandb.init(**cfg.wandb.setup, config=cfg)
    transform = default_transforms(cfg.dataset.decimate)
    train_ds = PointCloudData(cfg.dataset.data_root, transform)
    valid_ds = PointCloudData(
        cfg.dataset.data_root, transform, valid=True, folder="test"
    )
    train_loader = DataLoader(
        dataset=train_ds, batch_size=cfg.training.batch_size, shuffle=True
    )
    valid_loader = DataLoader(dataset=valid_ds, batch_size=cfg.training.batch_size)

    model = instantiate(cfg.model.model).to(device)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    pointnetloss = instantiate(cfg.training.loss)
    wandb.watch(model, **cfg.wandb.watch)
    print("setup complete")
    for _ in tqdm(range(cfg.training.epochs), desc="Epoch", leave=True):
        train(model, train_loader, optimizer, pointnetloss)
        val_acc = validation(model, valid_loader, cfg.dataset.decimate)
        if val_acc > 95:
            break


if __name__ == "__main__":
    wandb.login()
    main()
    wandb.finish()
