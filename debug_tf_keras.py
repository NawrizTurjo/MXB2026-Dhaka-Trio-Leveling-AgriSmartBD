
import os
# Force legacy consistency
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tf_keras
import tensorflow as tf

print(f"TF Version: {tf.__version__}")
print(f"tf_keras Version: {tf_keras.__version__}")

model_path = "./plant_disease_model/plant_disease_efficientnetb4.h5"

try:
    print("Loading model via tf_keras...")
    model = tf_keras.models.load_model(model_path, compile=False)
    print("SUCCESS: Model loaded with tf_keras!")
    model.summary()
except Exception as e:
    print(f"FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
