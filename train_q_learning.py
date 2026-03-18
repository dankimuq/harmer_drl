import numpy as np
import pickle
import os
from pentest_env import PentestEnv

def get_state_key(state_array):
    # Convert numpy array to integer tuple to use as dictionary key
    return tuple(state_array.tolist())

def train_q_learning(episodes=10000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995):
    env = PentestEnv()
    
    # Initialize Q-table
    # Since state space can be dynamic, we use a dictionary
    q_table = {}
    
    print("Starting Q-Learning Training...")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = get_state_key(obs)
        if state not in q_table:
            q_table[state] = np.zeros(env.action_space.n)
            
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore
            else:
                action = np.argmax(q_table[state]) # Exploit
                
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = get_state_key(next_obs)
            
            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_space.n)
            
            # Q-learning update rule
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error
            
            state = next_state
            total_reward += reward
            
        epsilon = max(0.01, epsilon * epsilon_decay)
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode: {episode + 1}, Epsilon: {epsilon:.3f}, Total Reward: {total_reward:.2f}")

    print("Training finished!")
    
    # Save Q-table
    os.makedirs("models", exist_ok=True)
    with open("models/q_table_agent.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Q-table saved to models/q_table_agent.pkl")
    
    # Evaluation
    evaluate_q_learning(env, q_table)

def evaluate_q_learning(env, q_table, eval_episodes=10):
    print("Evaluating Trained Q-Learning Policy...")
    total_rewards = []
    
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        state = get_state_key(obs)
        done = False
        rewards = 0
        
        while not done:
            if state in q_table:
                action = np.argmax(q_table[state])
            else:
                action = env.action_space.sample() # Fallback
                
            obs, reward, terminated, truncated, _ = env.step(action)
            state = get_state_key(obs)
            done = terminated or truncated
            rewards += reward
            
        total_rewards.append(rewards)
        
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"Mean reward (Q-Learning) over {eval_episodes} episodes: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    train_q_learning()
