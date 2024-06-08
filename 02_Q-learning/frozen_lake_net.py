import time;
#import gym; 
import gymnasium as gym;
import numpy as np;
import random;
import time;
import matplotlib.pyplot as plt;
from matplotlib.animation import FuncAnimation;
import tensorflow as tf;

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="rgb_array");
#env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="human");

lr = 0.5; #learning rate
y = 0.9; #discount factor lambda

embedding_dim = 4;
model = tf.keras.models.Sequential(
        [tf.keras.layers.Embedding(input_dim=env.observation_space.n, output_dim=embedding_dim),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(env.action_space.n)]);
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01);
model.compile(optimizer=optimizer, loss='mse');

for idx in range(100):
    observation, info = env.reset();
    print("eps", idx);
    while True:
        input_feature = np.array([[observation]]);
        predictions = model.predict(input_feature);
        if random.random() > 0.3:
            action = predictions[0].argmax();
        else:
            action = env.action_space.sample();
        observation_, reward, terminated, truncated, info = env.step(action);
        done = terminated;
        r_ = -1;
        if done:
            if reward == 1.0:
                r_ = 100000; # win
            else:
                r_ = -100000; # lost
        if observation_ == observation:
            r_ = -100;
        # scalered
        r_ = r_ / 10000.0;
        input_feature_ = np.array([[observation_]]);
        predictions_ = model.predict(input_feature_);
        predictions[0][action] = predictions[0][action] + lr * (r_ + y * np.max(predictions_[0]) - predictions[0][action]);
        #
        model.fit(input_feature, predictions, epochs=1);
        #
        observation = observation_;
        if done:
            break;
#
images = [];
def add_image(images):
    image = env.render();
    images.append(image);
#
WIN = False;
observation, info = env.reset();
add_image(images);

while True:
    input_feature = np.array([[observation]]);
    predictions = model.predict(input_feature);
    action = predictions[0].argmax();
    observation, reward, terminated, truncated, info = env.step(action);
    add_image(images);
    if terminated or truncated:
        if reward != 0.:
            WIN = True;
        break;
env.close();
print("Result(WIN):", WIN);

#show
print("len(images)", len(images));
fig, ax = plt.subplots();
im = ax.imshow(images[0], animated=True)
def update(frame):
    im.set_array(images[frame])
    return im,
ani = FuncAnimation(fig, update, frames=len(images), blit=True)
plt.show()

