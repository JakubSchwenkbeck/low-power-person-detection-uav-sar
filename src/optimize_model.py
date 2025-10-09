import tensorflow as tf
import os


def optimize_model(model_name: str, model_path: str, output_dir: str = "models/optimized_models"):
    """Calls float16 and int8 quantization"""
    optimize_model_float16(model_name, model_path, output_dir)
    optimize_model_int8(model_name, model_path, output_dir)


def optimize_model_float16(model_name: str, model_path: str, output_dir: str = "models/optimized_models"):
    """Float16 quantizations"""
    os.makedirs(output_dir, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    output_path = os.path.join(output_dir, f"{model_name}_float16.tflite")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model saved: {output_path} ({len(tflite_model) / (1024*1024):.2f} MB)")
    return output_path


def optimize_model_int8(model_name: str, model_path: str, output_dir: str = "models/optimized_models"):
    """Dynamic range quantization - weights only"""
    os.makedirs(output_dir, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    output_path = os.path.join(output_dir, f"{model_name}_dynamic.tflite")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model saved: {output_path} ({len(tflite_model) / (1024*1024):.2f} MB)")
    return output_path


