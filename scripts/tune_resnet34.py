import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from trashcan_core.components.data_loader import CustomDataLoader
from trashcan_core.components.constants import MATERIAL_VERSION_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 17
BATCH_SIZE = 32
INPUT_SIZE = (256, 256)


def get_pretrained_unet():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model


def fine_tune(model, train_loader, val_loader, epochs=10):
    for param in model.encoder.parameters():
        param.requires_grad = False

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader)}"
        )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader)}")

    print("Finished Training!")
    return model


def save_model_as_onnx(model, filename="models/resunet34v8.onnx"):
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


def main():
    data_loader = CustomDataLoader(MATERIAL_VERSION_PATH, batch_size=BATCH_SIZE)
    data_loader.crop(*INPUT_SIZE)

    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()

    model = get_pretrained_unet()

    model = fine_tune(model, train_loader, val_loader, epochs=10)

    save_model_as_onnx(model)


if __name__ == "__main__":
    main()
