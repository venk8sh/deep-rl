# Lunar Lander Deep Reinforcement Learning Agent 🚀

This notebook demonstrates how to train a Deep Reinforcement Learning (DRL) agent to land a Lunar Lander correctly on the Moon using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) and upload the trained agent to the Hugging Face Hub.

## Table of Contents
1. [Project Objective](#project-objective)
2. [Environment](#environment)
3. [Libraries Used](#libraries-used)
4. [Setup and Installation](#setup-and-installation)
5. [Training the Agent](#training-the-agent)
6. [Evaluating the Agent](#evaluating-the-agent)
7. [Pushing to Hugging Face Hub](#pushing-to-hugging-face-hub)
8. [Loading a Pre-trained Model](#loading-a-pre-trained-model)

## Project Objective

At the end of this notebook, you will be able to:
- Use **Gymnasium**, the environment library.
- Use **Stable-Baselines3**, the deep reinforcement learning library.
- Push your trained agent to the Hugging Face Hub with a video replay and an evaluation score.

## Environment

The environment used is [LunarLander-v2](https://gymnasium.farama.org/environments/box2d/lunar_lander/). The goal of the agent is to land the lander safely on the landing pad. The observation space is an 8-dimensional vector, and the action space is discrete with 4 actions: do nothing, fire left engine, fire main engine, fire right engine.

## Libraries Used
- `gymnasium[box2d]`: For the LunarLander-v2 environment.
- `stable-baselines3[extra]`: The deep reinforcement learning library (PPO algorithm).
- `huggingface_sb3`: For loading and uploading models from the Hugging Face Hub.
- `pyvirtualdisplay`: For creating a virtual screen to render environments in Colab.

## Setup and Installation

### 1. Set the GPU
Ensure your Colab runtime is set to GPU for faster training. Go to `Runtime > Change Runtime type` and select `GPU` as the `Hardware Accelerator`.

### 2. Install Dependencies
Run the following commands to install the required system packages and Python libraries. Note that due to Python 3.12 compatibility issues with `pygame` in Colab, specific versions are installed.

```python
!apt update && apt install -y swig cmake build-essential python3-dev
!pip uninstall -y pygame gymnasium box2d-py stable-baselines3
!pip install pygame==2.5.2
!pip install gymnasium[box2d]
!pip install stable-baselines3==2.0.0a5
!pip install swig huggingface_sb3
!pip install -r https://raw.githubusercontent.com/huggingface/deep-rl-class/main/notebooks/unit1/requirements-unit1.txt
```

### 3. Setup Virtual Display
A virtual display is needed to render the environment and record replay videos.

```python
!sudo apt-get update
!sudo apt-get install -y python3-opengl
!apt install ffmpeg
!apt install xvfb
!pip3 install pyvirtualdisplay

import os
os.kill(os.getpid(), 9) # Restart runtime to ensure new libraries are used

from pyvirtualdisplay import Display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
```

## Training the Agent

We use the PPO (Proximal Policy Optimization) algorithm from Stable-Baselines3 to train the agent. A vectorized environment with 16 instances is used to diversify training experiences.

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment
env = make_vec_env('LunarLander-v2', n_envs=16)

# Instantiate the PPO model with optimized parameters
model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1
)

# Train the agent for 1,000,000 timesteps
model_name = "ppo-LunarLander-v2"
model.learn(total_timesteps=1000000)
model.save(model_name)
```

## Evaluating the Agent

After training, the agent's performance is evaluated using `evaluate_policy` from Stable-Baselines3. The environment is wrapped with a `Monitor` to capture episode statistics.

```python
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

eval_env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
```

## Pushing to Hugging Face Hub

To share the trained model and track its performance on a leaderboard, it can be pushed to the Hugging Face Hub using `huggingface_sb3`.

1. **Create an account** on Hugging Face (if you don't have one): [huggingface.co/join](https://huggingface.co/join)
2. **Generate an authentication token** with a `write` role: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. **Log in** to your Hugging Face account in the notebook:

```python
from huggingface_hub import notebook_login
notebook_login()
!git config --global credential.helper store
```

4. **Package and push** the model:

```python
from huggingface_sb3 import package_to_hub
from stable_baselines3.common.vec_env import DummyVecEnv

env_id = "LunarLander-v2"
model_architecture = "PPO"
repo_id = "venky16067/ppo-LunarLander-v2" # REPLACE WITH YOUR HUGGING FACE USERNAME AND REPO NAME
commit_message = "Upload PPO LunarLander-v2 trained agent"

eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

package_to_hub(
    model=model,
    model_name=model_name,
    model_architecture=model_architecture,
    env_id=env_id,
    eval_env=eval_env,
    repo_id=repo_id,
    commit_message=commit_message
)
```

## Loading a Pre-trained Model

You can also load pre-trained models from the Hugging Face Hub.

```python
from huggingface_sb3 import load_from_hub

repo_id = "Classroom-workshop/assignment2-omar" # Example repo_id
filename = "ppo-LunarLander-v2.zip"

# Custom objects for compatibility if the model was trained with an older Python/SB3 version
custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}

checkpoint = load_from_hub(repo_id, filename)
model = PPO.load(checkpoint, custom_objects=custom_objects, print_system_info=True)

# Evaluate the loaded model
eval_env = Monitor(gym.make("LunarLander-v2"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
```
