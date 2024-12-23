from monai.losses import DiceLoss
import torch.optim as optim

from trashcan_core.components.data_loader import DataLoader
from trashcan_core.components.constants import MATERIAL_VERSION_PATH
from trashcan.components.models import ResUNetMini


def main():
    data_loader = DataLoader(MATERIAL_VERSION_PATH)

    model = ResUNetMini()

    loss = DiceLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    trainer = model.as_trainer(loss=loss, optimizer=optimizer, data_loader=data_loader)
    trainer.crop(height=128, width=128)

    trainer(30)

    trainer.save_model("models/miniv1.onnx", input_shape=(1, 3, 128, 128))


if __name__ == "__main__":
    main()
