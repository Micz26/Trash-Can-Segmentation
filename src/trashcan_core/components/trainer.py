from typing import Any
import torch.nn as nn
import torch
import torch.optim as optim
import onnx
import onnxruntime as ort

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

    def crop(self, height: int, width: int) -> None:
        def crop_tensor_batch(batch: torch.Tensor) -> torch.Tensor:
            original_height, original_width = batch.shape[-2], batch.shape[-1]

            if height > original_height or width > original_width:
                raise ValueError(
                    f"Target dimensions ({height}, {width}) must not exceed the original dimensions ({original_height}, {original_width})."
                )

            top = (original_height - height) // 2
            left = (original_width - width) // 2
            bottom = top + height
            right = left + width

            return batch[:, :, top:bottom, left:right]

        def crop_batches(batches: list) -> list:
            return [crop_tensor_batch(batch) for batch in batches]

        self.data_loader.x_train = crop_batches(self.data_loader.x_train)
        self.data_loader.y_train = crop_batches(self.data_loader.y_train)
        self.data_loader.x_val = crop_batches(self.data_loader.x_val)
        self.data_loader.y_val = crop_batches(self.data_loader.y_val)

        print(f"Cropped all dataset images to size ({height}, {width}).")

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
