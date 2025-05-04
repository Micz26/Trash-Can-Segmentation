# Trash Can Segmentation

## Description

This project focuses on segmenting trash cans in images using deep learning techniques. Two custom models based on the **ResUNet** architecture were developed:

- **Small ResUNet**: A lightweight model with fewer parameters, optimized for faster inference and lower resource consumption.
- **Large ResUNet**: A deeper network with increased capacity, aiming for improved accuracy.

Additionally, a pre-trained **ResNet** model from PyTorch's `torchvision.models` was fine-tuned to adapt to the trash can segmentation task, serving as a baseline for comparison.

## Approaches Used

- **Custom ResUNet Architectures**:
  - *Small Version*: Designed for environments with limited computational resources.
  - *Large Version*: Enhanced depth and complexity for higher segmentation accuracy.

- **Transfer Learning**:
  - Fine-tuning a pre-trained ResNet model to leverage existing feature extraction capabilities for the specific task of trash can segmentation.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Micz26/Trash-Can-Segmentation.git
   cd Trash-Can-Segmentation
   ```

2. **Create a Virtual Environment**:

   ```bash
   conda create --name trashcan-env python=3.9
   ```

3. **Activate the Virtual Environment**:

   ```bash
   conda activate trashcan-env
   ```

4. **Install Dependencies**:

   ```bash
   pip install .
   ```

5. **Download and Place the Dataset**:

   - Download the [TrashCan dataset](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7).
   - Extract and place the dataset into the `data/` directory.

## Usage

### Training

To train and save the model:

```bash
python scripts/train_resunet.py
```

### Docker Deployment

1. **Build the Docker Image**:

   ```bash
   docker build -t trashcan-app .
   ```

2. **Run the Docker Container**:

   ```bash
   docker run -p 8501:8501 trashcan-app
   ```

### Streamlit Interface (Development Mode)

To run the application using Streamlit:

```bash
streamlit run src/trashcan_frontend/frontend.py
```

**Note**: If you choose to run the application via Streamlit, ensure that you adjust the paths in the following file:

```python
src/trashcan_core/components/constants/file_paths.py
```
