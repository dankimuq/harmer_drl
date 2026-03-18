from stable_baselines3 import PPO, DQN, A2C
from pentest_env import PentestEnv
import pickle

env = PentestEnv()

for name, loader in [("PPO", lambda: PPO.load("models/ppo_pentest_agent", env=env)),
                     ("DQN", lambda: DQN.load("models/dqn_pentest_agent", env=env)),
                     ("A2C", lambda: A2C.load("models/a2c_pentest_agent", env=env))]:
    try:
        m = loader()
        obs, _ = env.reset()
        a, _ = m.predict(obs, deterministic=True)
        print(f"[OK] {name} loaded – sample action: {int(a)}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")

try:
    with open("models/q_table_agent.pkl", "rb") as f:
        qt = pickle.load(f)
    print(f"[OK] Q-table loaded – {len(qt)} states")
except Exception as e:
    print(f"[FAIL] Q-Learning: {e}")
