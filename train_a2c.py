import os
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from pentest_env import PentestEnv

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

def train_a2c():
    env = PentestEnv()
    
    print("Checking environment compatibility...")
    check_env(env)
    env = Monitor(env)
    
    print("Initializing A2C (Advantage Actor-Critic) agent...")
    # A2C is a synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C).
    # It strikes a good balance between the value-based DQN and the policy-gradient PPO.
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./a2c_pentest_tensorboard/")
    
    print("Training A2C started...")
    total_timesteps = 10000
    model.learn(total_timesteps=total_timesteps, tb_log_name="A2C_Pentest")
    print("Training finished!")
    
    model_path = f"{models_dir}/a2c_pentest_agent"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    
    print("Evaluating trained A2C policy...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward (A2C): {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    train_a2c()
