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

np.set_printoptions(threshold=sys.maxsize);
# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
#env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env = gym.make("ALE/Pong-v5", render_mode="human")
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

def preprocess(image):
    image = image[35:195]   
    image = image[::2,::2,0]
    image[image==144] = 0
    image[image==109] = 0
    image[image!=0] = 1.0
    #return image
    return image.astype('float').ravel()

model = load_model("./models/model.{}.h5".format(sys.argv[1]))
print(model.summary())

state, _ = env.reset()
state = preprocess(state)
while True:  # Run until solved
    state = ops.convert_to_tensor(state)
    state = ops.expand_dims(state, 0)

    # Predict action probabilities and estimated future rewards
    # from environment state
    action_probs, _ = model(state)

    # Sample action from action probability distribution
    action = np.random.choice(env.action_space.n, p=np.squeeze(action_probs))
    print(action, action_probs)

    # Apply the sampled action in our environment
    state, reward, done, _, _ = env.step(action)
    state = preprocess(state)

    if done:
        break
