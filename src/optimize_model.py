import tensorflow as tf
import os


def optimize_model(model_name: str, model_path: str, output_dir: str = "models/optimized_models"):
    """Generate different variants of optimized models"""
    

    os.makedirs(output_dir, exist_ok=True)
    
    # Size optimized variants 
    _convert(model_name, model_path, output_dir, "size_float32", tf.lite.Optimize.OPTIMIZE_FOR_SIZE, None)
    _convert(model_name, model_path, output_dir, "size_float16", tf.lite.Optimize.OPTIMIZE_FOR_SIZE, tf.float16)
    _convert(model_name, model_path, output_dir, "size_dynamic", tf.lite.Optimize.OPTIMIZE_FOR_SIZE, "dynamic")
    


    # Latency optimized variants
    _convert(model_name, model_path, output_dir, "latency_float32", tf.lite.Optimize.OPTIMIZE_FOR_LATENCY, None)
    _convert(model_name, model_path, output_dir, "latency_float16", tf.lite.Optimize.OPTIMIZE_FOR_LATENCY, tf.float16)
    _convert(model_name, model_path, output_dir, "latency_dynamic", tf.lite.Optimize.OPTIMIZE_FOR_LATENCY, "dynamic")




def _convert(model_name: str, model_path: str, output_dir: str, suffix: str, optimization, quant_type):
    """Helper function to convert model with specific settings"""


    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [optimization]

    # restrict to tensorflow builtin ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # float16 quantization
    if quant_type == tf.float16:
        converter.target_spec.supported_types = [tf.float16]
        converter.inference_input_type = tf.float16
        converter.inference_output_type = tf.float16
    # try dynamic range quantization
    elif quant_type == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    else:
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

    # try with experimental converter
    converter.experimental_new_converter = True

    # custom ops (not neccessary i think)
    converter.allow_custom_ops = True


    try:
        tflite_model = converter.convert()

    except ValueError as e:

        print(f"Skipping {suffix} for {model_name}: {e}")

        return None

    output_path = os.path.join(output_dir, f"{model_name}_{suffix}.tflite")

    with open(output_path, 'wb') as f:

        f.write(tflite_model)


    print(f"{suffix}: {output_path} ({len(tflite_model) / (1024*1024):.2f} MB)")


    return output_path


