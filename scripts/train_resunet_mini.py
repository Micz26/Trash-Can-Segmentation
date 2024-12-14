from monai.losses import DiceLoss
import torch.optim as optim

from trashcan_core.components.data_loader import DataLoader
from trashcan_core.components.constants import MATERIAL_VERSION_PATH
from trashcan.components.models import ResUNetMini


data_loader = DataLoader(MATERIAL_VERSION_PATH)

model = ResUNetMini()

loss = DiceLoss()

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

trainer = model.as_trainer(loss=loss, optimizer=optimizer, data_loader=data_loader)

trainer(1)

trainer.save_model("model.onnx", input_shape=(1, 3, 128, 128))
