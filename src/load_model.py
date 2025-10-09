# load_model.py
from ultralytics import YOLO
import tensorflow as tf
import os

def load_and_convert(model_name: str, model_type: str = "yolo"):
    """
    Loads a YOLO or FOMO model.
    If YOLO (.pt), exports to TensorFlow SavedModel format.
    If FOMO (.h5), just loads into memory.
    """
    os.makedirs("models", exist_ok=True)

    if model_type.lower() == "yolo":
        model = YOLO(model_name)
        model.export(format="saved_model")
        print("Saved under 'runs/export/weights/saved_model/'")

    elif model_type.lower() == "fomo":
        model = tf.keras.models.load_model(model_name)
        model.save("models/fomo_saved_model")
        print("Saved under 'models/fomo_saved_model/'")

    else:
        raise ValueError("model_type must be 'yolo' or 'fomo'")
