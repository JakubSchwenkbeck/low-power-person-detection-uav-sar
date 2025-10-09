import tensorflow as tf
import os


def optimize_model(model_name: str, model_path: str, output_dir: str = "models/optimized_models"):
    """
    Model optimization using TensorFlow Lite optimizations.
    
    Args:
        model_path: Path to the saved model (YOLO SavedModel format)
        output_dir: Directory to save optimized model
    
    Returns:
        Path to optimized TFLite model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
    # Apply optimizations --- from slides from lecutre 3
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert to TFLite
    tflite_model = converter.convert()
    
    output_path = os.path.join(output_dir, f"{model_name}_optimized_model.tflite")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Optimized model saved to: {output_path}")
    print(f"Model size: {len(tflite_model) / (1024*1024):.2f} MB")
    
    return output_path


