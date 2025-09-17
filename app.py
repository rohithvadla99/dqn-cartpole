# ------------------------------
# app.py
# ------------------------------

import streamlit as st
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import imageio
from io import BytesIO

# ------------------------------
# DQN Network
# ------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ------------------------------
# Session State
# ------------------------------
if 'model' not in st.session_state:
    st.session_state.model = None
if 'rewards' not in st.session_state:
    st.session_state.rewards = []

st.title("DQN CartPole Demo - Streamlit Cloud")

# ------------------------------
# Sidebar Parameters
# ------------------------------
episodes = st.sidebar.slider("Episodes", 100, 1000, 300, step=100)
epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.90, 0.999, 0.995, step=0.001)
lr = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")

# ------------------------------
# Train Agent Function
# ------------------------------
def train_agent(episodes, epsilon_decay, lr):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = DQN(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    buffer = deque(maxlen=10000)

    epsilon = 1.0
    epsilon_min = 0.01
    gamma = 0.99
    batch_size = 64

    rewards_per_episode = []

    for ep in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)
                
                q_values = model(states).gather(1, actions)
                next_q_values = model(next_states).max(1)[0].detach().unsqueeze(1)
                target_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                loss = loss_fn(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards_per_episode.append(total_reward)

    st.session_state.model = model
    st.session_state.rewards = rewards_per_episode

# ------------------------------
# Buttons
# ------------------------------
if st.button("Train Agent"):
    train_agent(episodes, epsilon_decay, lr)
    st.success("Training complete!")

# ------------------------------
# Plot rewards
# ------------------------------
if st.session_state.rewards:
    fig, ax = plt.subplots()
    ax.plot(st.session_state.rewards, label="Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN Training Progress")
    ax.legend()
    st.pyplot(fig)

# ------------------------------
# Run Agent Animation
# ------------------------------
if st.button("Run Trained Agent (Animation)"):
    if st.session_state.model is None:
        st.warning("No trained model found. Please train first.")
    else:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        state, info = env.reset()
        done = False
        frames = []

        # Collect frames as RGB arrays
        while not done:
            frame = env.render()
            frames.append(frame)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = torch.argmax(st.session_state.model(state_tensor)).item()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        env.close()
        st.success(f"Agent finished episode with reward: {len(frames)}")

        # Create GIF using imageio
        buf = BytesIO()
        imageio.mimsave(buf, frames, format='GIF', duration=0.05)
        st.image(buf.getvalue(), caption="Agent Animation")
