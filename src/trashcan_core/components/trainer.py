from typing import Any
import torch.nn as nn
import torch
import torch.optim as optim

from trashcan_core.components.data_loader import DataLoader


class Trainer:
    def __init__(
        self,
        net: nn.Module,
        loss: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
    ) -> None:
        self.net = net
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        self.loss = loss
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.load_data()

    def __call__(self, epochs: int) -> Any:
        for epoch in range(epochs):
            self.net.train()
            running_loss = 0.0

            for images, labels in zip(
                self.data_loader.x_train, self.data_loader.y_train
            ):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.net(images)
                loss = self.loss(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(
                f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(self.data_loader)}"
            )

            self.net.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in zip(
                    self.data_loader.x_val, self.data_loader.y_val
                ):
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.net(images)
                    loss = self.loss(outputs, labels)

                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    _, true = torch.max(labels, 1)
                    total += true.size(0)
                    correct += (predicted == true).sum().item()

            print(
                f"Validation Loss: {val_loss / len(self.data_loader)}, Accuracy: {100 * correct / total}%"
            )

        print("Finished Training")

    def load_data(self):
        self.data_loader()

    def get_data(self):
        return (
            self.data_loader.x_train,
            self.data_loader.y_train,
            self.data_loader.x_val,
            self.data_loader.y_val,
        )

    def get_model(self):
        return self.net
