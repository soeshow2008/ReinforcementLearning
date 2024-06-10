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
    #return image
    return image.astype('float').ravel()

# Actor-Critiic, TD
# cnn
#inputs = layers.Input(shape=(80, 80, 1)) 
#common = Conv2D(4, (3, 3), activation='relu')(inputs)
#common = MaxPooling2D(pool_size=(2, 2))(common)
#common = Conv2D(8, (3, 3), activation='relu')(common)
#common = MaxPooling2D(pool_size=(2, 2))(common)
#common = Flatten()(common);

# liner
inputs = layers.Input(shape=(80*80,))
common = layers.Dense(128, activation="relu")(inputs)
common = layers.Dense(64, activation="relu")(common)
action = layers.Dense(env.action_space.n, activation="softmax")(common)
critic = layers.Dense(1)(common)
model = keras.Model(inputs=inputs, outputs=[action, critic])
model = load_model("./models/model.base.h5")
#model = load_model("./models/model.{}.h5".format(sys.argv[1]))
# Train
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

print(model.summary())

while True:  # Run until solved
    episode_begin = True
    state, _ = env.reset()
    state = preprocess(state)
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # env.render() Adding this :%line would show the attempts
            # of the agent in a pop up window.

            state = ops.convert_to_tensor(state)
            state = ops.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(env.action_space.n, p=np.squeeze(action_probs))
            if episode_begin and episode_count % 5 == 0:
                print(action, action_probs)
            action_probs_history.append(ops.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, _, _ = env.step(action)
            state = preprocess(state)

            rewards_history.append(reward)
            episode_reward += reward
            episode_begin = False
            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up receiving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(ops.expand_dims(value, 0), ops.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 5 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count), flush=True)
    
    if episode_count % 50 == 3:
        print("save model {}!".format(episode_count), flush=True)
        model.save("./models/model.{}.h5".format(episode_count))
    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count), flush=True)
        break
