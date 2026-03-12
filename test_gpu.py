import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Visualization & Metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Performance / device setup
print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
USE_GPU = len(gpus) > 0
if USE_GPU:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("GPUs:", gpus)
else:
    print("GPUs: none (TensorFlow will use CPU).")
    print("Tip: On Windows, TensorFlow GPU is typically available via WSL2 (Ubuntu) + NVIDIA drivers.")

# XLA (jit_compile) usually helps on GPU
USE_JIT = USE_GPU
print("Successfully tested GPU setup script.")
