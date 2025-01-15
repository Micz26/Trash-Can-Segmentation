from monai.losses import DiceLoss
import torch.optim as optim


from trashcan_core.components.data_loader import CustomDataLoader
from trashcan_core.components.constants import MATERIAL_VERSION_PATH
from trashcan.components.models import ResUNetMini
from trashcan_core.components.trainer import Trainer


def main():
    data_loader = CustomDataLoader(MATERIAL_VERSION_PATH, batch_size=32, shuffle=True)
    data_loader.crop(256, 256)

    model = ResUNetMini()
    loss = DiceLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    trainer = Trainer(
        net=model,
        loss=loss,
        optimizer=optimizer,
        data_loader=data_loader,
    )

    trainer(epochs=1)

    trainer.save_model("models/resunet_small.onnx", input_shape=(1, 3, 256, 256))


if __name__ == "__main__":
    main()
