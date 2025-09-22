import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys

MODEL_PATH = 'plant_disease_model.h5'
CLASS_INDICES_PATH = 'plant_disease_class_indices.npy'
IMG_SIZE = (128, 128)

def predict(img_path):
    model = load_model(MODEL_PATH)
    class_indices = np.load(CLASS_INDICES_PATH, allow_pickle=True).item()
    idx_to_class = {v: k for k, v in class_indices.items()}
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_class = idx_to_class[pred_idx]
    return pred_class

if __name__ == "__main__":
    img_path = sys.argv[1]
    pred = predict(img_path)
    print(f"Predicted class: {pred}")
