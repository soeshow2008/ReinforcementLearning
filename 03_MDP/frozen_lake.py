import time;
#import gym; 
import gymnasium as gym; 
import numpy as np;
import random;
import time;
import matplotlib.pyplot as plt;
from matplotlib.animation import FuncAnimation;

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="rgb_array");

#
U = np.zeros([env.observation_space.n]);
# init 
termS = set();
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        for p, s_, r, done in env.env.P[s][a]:
            if done:
                termS.add(s_);
#
for s in range(env.observation_space.n):
    if s in termS:
        continue;
    for a in range(env.action_space.n):
        for p, s_, r, done in env.env.P[s][a]:
            if not done:
                continue;
            if r == 0:
                U[s_] = -100000;
            if r == 1:
                U[s_] = 100000;
#
print("termS", termS);
print("init U", U);

y = 0.8; #discount factor lambda
eps = 1000;

for i in range(eps):
    prev_U = np.copy(U);
    for s in range(env.observation_space.n):
        if s in termS:
            continue;
        u_s = -100000;
        for a in range(env.action_space.n):
            q_sa = 0.0; # Q-value
            for p, s_, r, _ in env.env.P[s][a]:
                q_sa += p * (r + y * prev_U[s_]);
            u_s = max(u_s, q_sa);
        #
        U[s] = u_s;
print(U);
#
images = [];
def add_image(images):
    image = env.render();
    images.append(image);
#
WIN = False;
observation, info = env.reset();
add_image(images);

s = 0;
while True:
    action = -1;
    u_s = -100000;
    for a in range(env.action_space.n):
        p, s_, r, _ = env.env.P[s][a][0];
        if action == -1 or U[s_] > u_s:
            u_s = U[s_];
            action = a;
    observation, reward, terminated, truncated, info = env.step(action);
    s = observation;
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

