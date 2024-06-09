import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import ops
from keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
import gymnasium as gym 
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
#env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env = gym.make("ALE/Pong-v5", render_mode="human")
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

# Actor-Critiic, TD
inputs = layers.Input(shape=env.observation_space.shape) 
common = Conv2D(32, (3, 3), activation='relu')(inputs)
common = MaxPooling2D(pool_size=(2, 2))(common)
common = Flatten()(common);
action = layers.Dense(env.action_space.n, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

model = keras.Model(inputs=inputs, outputs=[action, critic])
model = load_model("./models/model.{}.h5".format(sys.argv[1]))
#
state, _ = env.reset()
state = state / 255.0
while True:  # Run until solved
    state = ops.convert_to_tensor(state)
    state = ops.expand_dims(state, 0)

    # Predict action probabilities and estimated future rewards
    # from environment state
    action_probs, _ = model(state)

    # Sample action from action probability distribution
    action = np.random.choice(env.action_space.n, p=np.squeeze(action_probs))
    print(action_probs, action)

    # Apply the sampled action in our environment
    state, reward, done, _, _ = env.step(action)
    state = state / 255.0
    if done:
        break
