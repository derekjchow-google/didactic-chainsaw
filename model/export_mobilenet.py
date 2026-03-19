# /// script
# dependencies = [
#   "tensorflow",
# ]
# ///

import tensorflow as tf
import os

def representative_data_gen():
    # Generate dummy data for the representative dataset
    for _ in range(100):
        # MobileNetV1 expects input in range [-1, 1]
        data = tf.random.uniform((1, 224, 224, 3), minval=-1, maxval=1, dtype=tf.float32)
        yield [data]

def export():
    # Load MobileNetV1 pre-trained on ImageNet
    base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Extract the first three convolutions (up to conv_pw_1_relu)
    layer_name = 'conv_pw_1_relu'
    output = base_model.get_layer(layer_name).output
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    
    # Convert to TFLite with Full Integer Quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 or int8 (optional, but common for "quantized" models)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Ensure the directory exists
    os.makedirs('model', exist_ok=True)
    
    # Save the TFLite model
    output_path = 'model/mobilenet_first_3_convs_quant.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Successfully exported the quantized first three convolutions to {output_path}")

if __name__ == "__main__":
    export()
