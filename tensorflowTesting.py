import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define Sequential model with 3 layers
model = keras.Sequential([
    layers.Dense(2, activation="relu", name="layer1", use_bias=False),
    layers.Dense(3, activation="relu", name="layer2", use_bias=False),
    layers.Dense(1, name="layer3"),
])
# Call model on a test input
x = tf.ones((1, 3))
print(x)
y = model(x)
print(y)
model.summary()