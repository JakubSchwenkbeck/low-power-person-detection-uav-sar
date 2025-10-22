import tensorflow as tf
import os


def optimize_model(
    model_name: str,
    model_path: str,
    output_dir: str = "models/optimized_models",
    
):
    """Generate optimized TFLite variants and optionally verify sizes.
   """
  

    os.makedirs(output_dir, exist_ok=True)

    # f32 ( baseline )
    fp32_default = _convert(
        model_name, model_path, output_dir, "float32_default", tf.lite.Optimize.DEFAULT, None, use_sparsity=False
    )
    fp32_experimental = _convert(
        model_name, model_path, output_dir, "float32_experimental", tf.lite.Optimize.DEFAULT, None, use_sparsity=True
    )

    # Float16 (but IO f32)
    fp16_default = _convert(
        model_name, model_path, output_dir, "float16_default", tf.lite.Optimize.DEFAULT, tf.float16, use_sparsity=False
    )
    fp16_experimental = _convert(
        model_name, model_path, output_dir, "float16_experimental", tf.lite.Optimize.DEFAULT, tf.float16, use_sparsity=True
    )

    # Dynamic range int8 (weights int8, float32 IO)
    dyn_int8_default = _convert(
        model_name, model_path, output_dir, "dynamic_int8_default", tf.lite.Optimize.DEFAULT, "dynamic", use_sparsity=False
    )
    dyn_int8_experimental = _convert(
        model_name, model_path, output_dir, "dynamic_int8_experimental", tf.lite.Optimize.DEFAULT, "dynamic", use_sparsity=True
    )

  



def _convert(model_name: str, model_path: str, output_dir: str, suffix: str, optimization, quant_type, use_sparsity: bool = False):
    """Helper to convert a model with specific settings.

    If use_sparsity=True we include EXPERIMENTAL_SPARSITY with DEFAULT so that sparse
    weights ( if they are present ) are encoded more efficiently. This does not perform pruning.
    """


    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)


    optimizations = [optimization] if optimization else [tf.lite.Optimize.DEFAULT]


    if use_sparsity and tf.lite.Optimize.EXPERIMENTAL_SPARSITY not in optimizations:
        # Combine DEFAULT + EXPERIMENTAL_SPARSITY when requested.
        optimizations.append(tf.lite.Optimize.EXPERIMENTAL_SPARSITY)


    converter.optimizations = optimizations

    # restrict to tensorflow builtin ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # float16 quantization
    if quant_type == tf.float16:
        converter.target_spec.supported_types = [tf.float16]
    
    elif quant_type == "dynamic":
        # Dynamic range quantization always uses DEFAULT; include sparsity if requested.
        converter.optimizations = [tf.lite.Optimize.DEFAULT] + ([tf.lite.Optimize.EXPERIMENTAL_SPARSITY] if use_sparsity else [])
    
    # try with experimental converter
    converter.experimental_new_converter = True


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


