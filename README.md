# Trash Can Segmentation

1. Clone repo

```bash
git clone https://github.com/Micz26/CNN-Waste-Detector.git
```

2. Create venv

```bash
conda create --name venv
```

3. Activate venv

```bash
conda activate venv
```

4. Activate venv

```bash
pip install .
```

5. Add [data](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7) to data/

6. train and save

```
python scripts/train_resunet.py
```

7. Build docker app

```
docker build -t trashcan-app .
```

8. Run app

```
docker run -p 8501:8501 trashcan-app
```

9. Run using streamlit

```
streamlit run src/trashcan_frontend/frontend.py
```

- **Note**: If you want to run using Streamlit, you must adjust the paths in: `src/trashcan_core/components/constants/file_paths.py`.
