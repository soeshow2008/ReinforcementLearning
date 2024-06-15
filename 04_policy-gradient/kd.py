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
from scipy import sparse
import random
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.set_printoptions(threshold=sys.maxsize);
# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
#env = gym.make("ALE/Pong-v5", render_mode="human")
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

def preprocess(image):
    image = image[35:195]   
    image = image[::2,::2,0]
    image[image==144] = 0
    image[image==109] = 0
    image[image!=0] = 1.0
    flatten_image = image.astype('float').ravel()
    return flatten_image, image

teacher_model = load_model("./models/model.{}.h5".format(sys.argv[1]))

# Actor-Critiic, TD
# cnn
inputs = layers.Input(shape=(80, 80, 1)) 
common = Conv2D(32, (3, 3), activation='relu')(inputs)
common = MaxPooling2D(pool_size=(2, 2))(common)
common = Conv2D(64, (3, 3), activation='relu')(common)
common = MaxPooling2D(pool_size=(2, 2))(common)
common = Flatten()(common);
common = layers.Dense(128, activation="relu")(common)
#
action = layers.Dense(64, activation="relu")(common)
action = layers.Dense(env.action_space.n, activation="softmax", name='action')(action)
critic = layers.Dense(64, activation="relu")(common)
critic = layers.Dense(1, name='critic')(critic)
model = keras.Model(inputs=inputs, outputs=[action, critic])
model.compile(optimizer='adam',
              loss={'action': 'categorical_crossentropy', 
                    'critic': 'binary_crossentropy'})
print(model.summary())

episode_count = 0
state, _ = env.reset()
state, state2 = preprocess(state)
#
state2_replays = []
action_probs_replays = []
critic_value_replays = []
#
do_random_action = False
while True:  # Run until solved
    state = ops.convert_to_tensor(state)
    state = ops.expand_dims(state, 0)
    #
    state2 = ops.convert_to_tensor(state2)
    state2 = tf.reshape(state2, [1, 80, 80, 1])

    # Predict action probabilities and estimated future rewards
    # from environment state
    action_probs, critic_value = teacher_model(state)

    # train
    state2_replays.append(state2)
    action_probs_replays.append(action_probs)
    critic_value_replays.append(critic_value)

    # Sample action from action probability distribution
    if do_random_action and random.random() > 0.9:
        action = env.action_space.sample()
    else:
        action = np.random.choice(env.action_space.n, p=np.squeeze(action_probs))

    # Apply the sampled action in our environment
    state, reward, done, _, _ = env.step(action)
    state, state2 = preprocess(state)

    if done:
        state2s = tf.concat(state2_replays, axis=0)
        action_probss = tf.concat(action_probs_replays, axis=0)
        critic_values = tf.concat(critic_value_replays, axis=0)
        model.fit(state2s, {'action': action_probss, 'critic': critic_values}, epochs=1)
        state2_replays.clear()
        action_probs_replays.clear()
        critic_value_replays.clear()
        #
        episode_count += 1
        if episode_count % 50 == 3:
            print("save model {}!".format(episode_count), flush=True)
            model.save("./student_models/model.{}.h5".format(episode_count))
        #
        state, _ = env.reset()
        state, state2 = preprocess(state)
        do_random_action = False
        if random.random() > 0.7:
            do_random_action = True
