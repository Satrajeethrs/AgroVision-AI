
# Plant Disease Classification — Deep Learning Project

This project provides a deep learning solution for classifying plant diseases from leaf images using a convolutional neural network (CNN). It uses a large, labeled image dataset and Keras/TensorFlow for model training and inference.

## Project Structure

- `archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/` — Training images, organized by class
- `archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/` — Validation images, organized by class
- `train_plant_disease_model.py` — Main script for training the CNN model
- `plant_disease_model_final.h5` — Saved trained model (Keras HDF5 format)
- `plant_disease_class_indices.npy` — Mapping of class names to indices
- `requirements.txt` — Python dependencies

## Dataset

The dataset consists of tens of thousands of plant leaf images, each labeled by disease and plant type. The images are organized in subfolders by class:

- `archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/<class_name>/`
- `archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/<class_name>/`

Each `<class_name>` is a unique plant/disease combination (e.g., `Apple___Apple_scab`, `Tomato___Early_blight`).



## Quick Start: Run Files in Order

Follow these steps to set up, train, and use all models in this project (plant disease, crop recommendation, and fertilizer recommendation). Run each file in the order shown:

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

- `requirements.txt` — Contains all required Python packages (TensorFlow, numpy, scikit-learn, Flask, etc.)

### 3. (Optional) Prepare or verify datasets

- Ensure your plant disease dataset is in:
   - `archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/`
   - `archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/`
- (If you have a script like `prepare_plant_disease_dataset.py`, run it to preprocess or verify the dataset.)
- For crop/fertilizer models, ensure any required CSVs (e.g., `crop_data.csv`, `fertilizer.csv`) are present.

### 4. Train or use the Plant Disease Classification Model

```bash
python train_plant_disease_model.py
```

- `train_plant_disease_model.py` — Loads the dataset, builds the CNN, and trains the model. Outputs:
   - `plant_disease_model_best.h5` — Best model (by validation accuracy)
   - `plant_disease_model_final.h5` — Final model after all epochs
   - `plant_disease_class_indices.npy` — Class label mapping

### 5. Train or use the Crop and Fertilizer Recommendation Models

- If you need to retrain the crop or fertilizer models:
   - `train_model.py` — Trains the crop and/or fertilizer recommendation models using tabular data (e.g., `crop_data.csv`, `fertilizer.csv`). Produces:
      - `model.pkl` — Trained crop recommendation model
      - `scaler.pkl` — Feature scaler for input normalization

- If models already exist, you can skip retraining and use them directly.

### 6. Run the Web Application (Crop & Fertilizer Recommendation UI)

```bash
python app.py
```

- `app.py` — Flask web application for crop and fertilizer recommendations. Loads `model.pkl` and `scaler.pkl` for predictions.

### 7. (Optional) Validate Recommendations with LLM

- `llm_validator.py` — Validates recommendations using an LLM provider or stub.
- `validate_recs.py` — CLI tool to validate and print recommendations.

Example:
```bash
python validate_recs.py
```

### 8. (Optional) Evaluate or test the models

- You can add or run scripts (e.g., `test_plant_disease_model.py`, `comprehensive_test.py`) to evaluate the models on test data or with your own samples.

Example:
```bash
python test_plant_disease_model.py
python comprehensive_test.py
```

### 9. Model Inference (Manual usage)

You can load and use the trained models in any Python environment:

**Plant Disease Model:**
```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('plant_disease_model_final.h5')
class_indices = np.load('plant_disease_class_indices.npy', allow_pickle=True).item()
# Preprocess your image and use model.predict(...)
```

**Crop Recommendation Model:**
```python
import joblib
import numpy as np

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
# X = scaler.transform([[your, input, features]])
# prediction = model.predict(X)
```

## Model Architecture

- Input: 128x128 RGB images
- 3 convolutional layers (32, 64, 128 filters)
- Max pooling after each conv layer
- Dense layer (256 units) + Dropout
- Output: Softmax over all classes

Optimizer: Adam (lr=0.0001)
Loss: Categorical crossentropy
Metrics: Accuracy
Epochs: 40 (configurable)

## Improving Accuracy

- The script uses data augmentation (horizontal flip, rotation)
- You can further tune:
  - Number of epochs (increase for better accuracy)
  - Model depth/width
  - Learning rate
  - Data augmentation parameters

## Testing and Evaluation

- The script prints validation accuracy after each epoch.
- For further evaluation, you can add a test set and use `model.evaluate(...)`.
- To improve results, consider:
  - Adding more layers or regularization
  - Using transfer learning (e.g., MobileNetV2, ResNet50)
  - Fine-tuning hyperparameters

## Troubleshooting

- If you see TensorFlow/Keras import errors, ensure your virtual environment is activated and dependencies are installed.
- If you run out of memory, reduce batch size or image size.
- Training may take significant time depending on hardware and dataset size.

## License

This project is for educational and research purposes. Dataset sources and licensing may apply; please review the dataset origin for any usage restrictions.# AgroVision-AI
