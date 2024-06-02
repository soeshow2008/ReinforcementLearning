import time;
import gym; 
import numpy as np;
import random;
import time;
import matplotlib.pyplot as plt;
from matplotlib.animation import FuncAnimation;
from gym.envs.toy_text.frozen_lake import generate_random_map;

#env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="rgb_array");
#env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="human");
env = gym.make('FrozenLake-v1', desc=generate_random_map(size=16), is_slippery=False, render_mode="rgb_array");


lr = 0.5; #learning rate
y = 0.9; #discount factor lambda

Q_table = np.random.uniform(0., 0.00001, [env.observation_space.n, env.action_space.n]);

def epsilon_greedy(Q_table, observation):
    p = random.random();
    if p > 0.3:
        return np.argmax(Q_table[observation, :]);
    else:
        return env.action_space.sample();

for idx in range(10000):
    observation, info = env.reset();
    last_action = -1;
    if idx % 1000 == 1:
        print("eps", idx);
    while True:
        action = epsilon_greedy(Q_table, observation);
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
        if r_ == -1 and last_action != -1 and last_action != action:
            r_ = -10;
        Q_table[observation, action] = Q_table[observation, action] + lr * (r_ + y * np.max(Q_table[observation_, :]) - Q_table[observation, action]);
        observation = observation_;
        last_action = action;
        if done:
            break;
print("Q_table", Q_table);

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
    action = np.argmax(Q_table[observation, :]);
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

