import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import segmentation_models_pytorch as smp

from trashcan_core.components.data_loader import DataLoader as CustomLoader
from trashcan_core.components.constants import MATERIAL_VERSION_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 17
BATCH_SIZE = 32
INPUT_SIZE = (128, 128)


def get_pretrained_unet():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model


def fine_tune(model, train_data, val_data, epochs=10):
    for param in model.encoder.parameters():
        param.requires_grad = False

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_data:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_data)}"
        )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_data:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(val_data)}")

    print("Finished Training!")
    return model


def preprocess_batches(x_batches, y_batches):
    """
    Converts a list of batches to PyTorch tensors.
    Assumes batches are lists of tensors with shape: [B, C, H, W].
    """
    x_tensor = torch.cat(x_batches, dim=0)
    y_tensor = torch.cat(y_batches, dim=0)
    return x_tensor, y_tensor


data_loader = CustomLoader(MATERIAL_VERSION_PATH)
data_loader()


def crop(loader, height: int, width: int) -> None:
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

    loader.x_train = crop_batches(loader.x_train)
    loader.y_train = crop_batches(loader.y_train)
    loader.x_val = crop_batches(loader.x_val)
    loader.y_val = crop_batches(loader.y_val)

    print(f"Cropped all dataset images to size ({height}, {width}).")


crop(data_loader, 128, 128)

x_train_tensor, y_train_tensor = preprocess_batches(
    data_loader.x_train, data_loader.y_train
)
x_val_tensor, y_val_tensor = preprocess_batches(data_loader.x_val, data_loader.y_val)

train_loader = DataLoader(
    TensorDataset(x_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(x_val_tensor, y_val_tensor), batch_size=BATCH_SIZE
)

model = get_pretrained_unet()
model = fine_tune(model, train_loader, val_loader, epochs=10)


def save_model_as_onnx(model, filename="models/resunet34v4.onnx"):
    dummy_input = torch.randn(1, 3, *INPUT_SIZE).to(DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Model saved as {filename}")


save_model_as_onnx(model)
