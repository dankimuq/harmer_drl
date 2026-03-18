import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from pentest_env import PentestEnv

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

def train_dqn():
    env = PentestEnv()
    
    print("Checking environment compatibility...")
    check_env(env)
    env = Monitor(env)
    
    print("Initializing DQN (Deep Q-Network) agent...")
    # DQN is a powerful value-based SOTA algorithm for discrete action spaces
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_pentest_tensorboard/", learning_starts=1000)
    
    print("Training DQN started...")
    total_timesteps = 15000  # DQN sometimes needs slightly more steps to explore than PPO
    model.learn(total_timesteps=total_timesteps, tb_log_name="DQN_Pentest")
    print("Training finished!")
    
    model_path = f"{models_dir}/dqn_pentest_agent"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    
    print("Evaluating trained DQN policy...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward (DQN): {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    train_dqn()
