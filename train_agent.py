import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from pentest_env import PentestEnv

# Directory to save models
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

def train():
    # Intialize environment
    env = PentestEnv()
    
    # Check if the environment follows Stable Baselines 3 standards
    print("Checking environment compatibility...")
    check_env(env)
    
    # Wrap environment with Monitor for better logging
    env = Monitor(env)
    
    print("Initializing PPO agent...")
    # Initialize PPO Agent. 
    # MlpPolicy is a standard feed-forward neural network
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_pentest_tensorboard/")
    
    # Train the agent
    print("Training started...")
    total_timesteps = 10000
    model.learn(total_timesteps=total_timesteps, tb_log_name="PPO_Pentest")
    print("Training finished!")
    
    # Save the model
    model_path = f"{models_dir}/ppo_pentest_agent"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    
    # Evaluate policy
    print("Evaluating trained policy...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    train()
