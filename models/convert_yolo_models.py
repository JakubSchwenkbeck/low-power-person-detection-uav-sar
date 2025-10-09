import os
from ultralytics import YOLO

def convert_model_to_tflite(model_name : str):
    """
    Convert a YOLO model to TFLite format and save it to models/tiny_models directory.
    
    Args:
        model_name (str): Name of the model file ( with or without .pt extension )
    
    Returns:
        str: Path to the converted TFLite model
    """

    if not model_name.endswith('.pt'):
        model_name += '.pt'
    
    tiny_models_dir = "models/tiny_models"
    os.makedirs(tiny_models_dir, exist_ok=True) 
    
    model_path = f"models/base_models/{model_name}"
    model = YOLO(model_path)
    
    model.export(format="tflite")
    
    # Move the exported file to tiny_models directory
    base_name = model_name.replace('.pt', '')
    tflite_filename = f"{base_name}_float32.tflite"
    final_path = os.path.join(tiny_models_dir, tflite_filename)
    
    if os.path.exists(tflite_filename):
        os.rename(tflite_filename, final_path)
    
    print(f"Model converted and saved to: {final_path}")
    
    # Load the exported TFLite model for verification
    tflite_model = YOLO(final_path)
    
    return final_path




if __name__ == "__main__":
    # Convert the yolo11n model
    converted_model_path = convert_model_to_tflite("yolo11n")
    
    # tflite_model = YOLO(converted_model_path)
    # results = tflite_model("https://ultralytics.com/images/bus.jpg")


