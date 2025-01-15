from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import onnxruntime as ort

from trashcan_core.components.data_loader import CustomDataLoader


class Trainer:
    def __init__(
        self,
        net: nn.Module,
        loss: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: CustomDataLoader,
    ) -> None:
        self.net = net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.loss = loss
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()

    def __call__(self, epochs: int) -> Any:
        for epoch in range(epochs):
            self.train_one_epoch(epoch, epochs)
            self.validate()

        print("Finished Training")

    def train_one_epoch(self, epoch: int, total_epochs: int) -> None:
        self.net.train()
        running_loss = 0.0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.net(images)
            loss = self.loss(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{total_epochs}], Loss: {running_loss / len(self.train_loader):.4f}"
        )

    def validate(self) -> None:
        self.net.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)
                loss = self.loss(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                _, true = torch.max(labels, 1)
                total += true.size(0)
                correct += (predicted == true).sum().item()

        print(
            f"Validation Loss: {val_loss / len(self.val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%"
        )

    def get_model(self) -> nn.Module:
        return self.net

    def save_model(self, save_path: str, input_shape: tuple) -> None:
        self.net.eval()
        dummy_input = torch.randn(*input_shape).to(self.device)
        torch.onnx.export(
            self.net,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"Model saved in ONNX format at {save_path}.")

    @staticmethod
    def load_model(onnx_path: str) -> ort.InferenceSession:
        session = ort.InferenceSession(onnx_path)
        print(f"Model loaded from {onnx_path}.")
        return session
