from paramiko import SSHClient
from pymetasploit3.msfrpc import MsfRpcClient
from scp import SCPClient

from configuration import (IS_REMOTE, REMOTE_HOST, RPC_PASSWORD, RPC_PORT,
                           RPC_USERNAME)

client = MsfRpcClient(RPC_PASSWORD,
                      username=RPC_USERNAME,
                      port=RPC_PORT,
                      server=REMOTE_HOST)
if IS_REMOTE:
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(hostname=REMOTE_HOST,
                username='ubuntu',
                key_filename='/home/lesliebinbin/add-kn.pem')
    scp = SCPClient(ssh.get_transport())
else:
    ssh = None
    scp = None
