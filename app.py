import streamlit as st
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os

# ------------------------------
# 1. Define DQN Network
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
# Streamlit UI
# ------------------------------
st.title("DQN CartPole Demo")
st.write("A Deep Q-Network agent learning to balance the CartPole environment.")

episodes = st.sidebar.slider("Episodes", 100, 1000, 500, step=100)
epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.90, 0.999, 0.995, step=0.001)
lr = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")


# ------------------------------
# 2. Train Agent
# ------------------------------
if st.button("Train Agent"):
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
        st.write(f"Episode {ep+1}/{episodes} - Reward: {total_reward} - Epsilon: {epsilon:.3f}")

    # Plot rewards
    fig, ax = plt.subplots()
    ax.plot(rewards_per_episode, label="Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN Training Progress on CartPole-v1")
    ax.legend()
    st.pyplot(fig)

    # Save model
    torch.save(model.state_dict(), "dqn_cartpole.pth")
    st.success("Training complete! Model saved as dqn_cartpole.pth")


# ------------------------------
# 3. Run Trained Agent
# ------------------------------
if st.button("Run Trained Agent"):
    env = gym.make("CartPole-v1")  # No rendering
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load("dqn_cartpole.pth"))
    model.eval()

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = torch.argmax(model(state_tensor)).item()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    st.success(f"Agent finished episode with reward: {total_reward}")


# ------------------------------
# 4. Record Video of Agent
# ------------------------------
if st.button("Record Agent Gameplay (MP4)"):
    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_dir=video_dir, name_prefix="dqn_cartpole")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load("dqn_cartpole.pth"))
    model.eval()

    state, info = env.reset()
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = torch.argmax(model(state_tensor)).item()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()

    latest_video = sorted(os.listdir(video_dir))[-1]
    video_path = os.path.join(video_dir, latest_video)

    st.video(video_path)
    st.success("Gameplay recorded and displayed above!")
