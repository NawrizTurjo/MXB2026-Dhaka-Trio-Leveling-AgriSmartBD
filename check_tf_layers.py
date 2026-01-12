
import tensorflow as tf
print(f"TF Version: {tf.__version__}")
try:
    print("Searching for TFOpLambda...")
    if hasattr(tf.keras.layers, 'TFOpLambda'):
        print("Found in tf.keras.layers!")
    else:
        print("Not found in tf.keras.layers")
        # List all for debug
        # print(dir(tf.keras.layers))
except Exception as e:
    print(e)
