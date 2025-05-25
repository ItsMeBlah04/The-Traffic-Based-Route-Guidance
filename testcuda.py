import tensorflow as tf
print("TF version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())
print(tf.config.list_physical_devices('GPU'))