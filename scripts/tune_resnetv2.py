import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import segmentation_models_pytorch as smp
from monai.losses import DiceLoss

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


def one_hot_encode(labels, num_classes):
    """
    Converts class indices in labels to one-hot encoding.
    labels should be of shape [batch_size, height, width] and values between [0, num_classes-1].
    """
    batch_size, height, width = labels.size()  # Ensure this is 3-dimensional (B, H, W)

    # Create a tensor of zeros with the shape [batch_size, num_classes, height, width]
    one_hot = torch.zeros(batch_size, num_classes, height, width, device=labels.device)

    # Use long indexing to set the correct class in each pixel location to 1
    one_hot = one_hot.scatter(
        1, labels.unsqueeze(1), 1
    )  # scatter along the channel dimension

    return one_hot


def fine_tune(model, train_data, val_data, epochs=10):
    # Freeze the encoder layers (except the last layers)
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.encoder.layer4.parameters():  # Unfreeze the last layers
        param.requires_grad = True

    # Move model to the device (GPU/CPU)
    model = model.to(DEVICE)

    # Use MONAI's DiceLoss for segmentation
    criterion = DiceLoss(to_onehot_y=False, softmax=True)  # DiceLoss with softmax
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Try a higher learning rate
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1
    )  # Learning rate scheduler

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Loop over the batches
        for x_batch, y_batch in train_data:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            # Since y_batch is already one-hot encoded, we don't need to process it
            outputs = model(x_batch)

            # Compute the loss
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update the learning rate

            train_loss += loss.item()

        # Print training loss
        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_data)}"
        )

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_data:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

                # Use the same logic for validation
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        # Print validation loss
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
model = fine_tune(model, train_loader, val_loader, epochs=5)


def save_model_as_onnx(model, filename="models/resunet34v2.onnx"):
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
