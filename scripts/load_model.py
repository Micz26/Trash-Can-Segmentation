import numpy as np

from trashcan_core.components.trainer import Trainer

session = Trainer.load_model("models//model.onnx")

input_data = np.random.randn(1, 3, 128, 128).astype(np.float32)

outputs = session.run(None, {"input": input_data})
print("Model outputs:", outputs)
