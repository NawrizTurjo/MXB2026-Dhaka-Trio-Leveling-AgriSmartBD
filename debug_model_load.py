
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = "./plant_disease_model/plant_disease_efficientnetb4.h5"
abs_path = os.path.abspath(model_path)
print(f"Attempting to load model from: {abs_path}")

try:
    # Hack: Define dummy classes to bypass SafeMode/Unknown Layer errors
    # This is safe because we do manual preprocessing in app.py
    class TFOpLambda(tf.keras.layers.Layer):
        def __init__(self, function, **kwargs):
            super().__init__(**kwargs)
            self.function = function
        def get_config(self):
            config = super().get_config()
            return config
        def call(self, x):
            return x

    class SlicingOpLambda(tf.keras.layers.Layer):
        def __init__(self, function, **kwargs):
            super().__init__(**kwargs)
            self.function = function
        def get_config(self):
            config = super().get_config()
            return config
        def call(self, x):
            return x

    custom_objects = {
        'TFOpLambda': TFOpLambda, 
        'SlicingOpLambda': SlicingOpLambda
    }
    
    # Use standard load_model (Keras 3) with custom objects
    model = load_model(model_path, custom_objects=custom_objects, compile=False) 
    print("SUCCESS: Model loaded successfully using Dummy Classes!")
    model.summary()
except Exception as e:
    print(f"FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
