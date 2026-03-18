import time
import uuid

import yaml

from data_processing import initialize_targets_config

with open('attack-config.yml', 'r') as f:
    attack_config = yaml.safe_load(f)

SESSION_SUCCESSFULLY_OPEND = str(uuid.uuid4())
SESSION_FAIL_TO_OPEN = str(uuid.uuid4())

REMOTE_RESOURCE_DIRECTORY = attack_config['metasploit']['remote-dir']
REMOTE_HOST = attack_config['metasploit']['host']
TOKEN = attack_config['metasploit']['token']
WAIT_FOR_DATA_TIME_OUT = attack_config['metasploit']['attack-timeout']
START_TIMESTAMP = time.time()
APPROACH_NAME, TARGETS, PATH_FILE = initialize_targets_config(
    "target-config.json")
RPC_USERNAME = attack_config['metasploit']['rpc']['username']
RPC_PORT = attack_config['metasploit']['rpc']['port']
RPC_PASSWORD = attack_config['metasploit']['rpc']['password']
IS_REMOTE = attack_config['metasploit'].get('is-remote')
if IS_REMOTE is None:
    IS_REMOTE = True
