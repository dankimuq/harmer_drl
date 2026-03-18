import json
import time
import uuid

import requests
from graphviz import Digraph

from basic_usage import get_current_host_ip
from configuration import REMOTE_HOST, TOKEN


class ReportUtil:
    def __init__(self,
                 base_url=f"https://{REMOTE_HOST}:5443/api/v1/",
                 token=None):
        self._token = token
        self._base_url = base_url

    def generate_unique_work_space(self):
        return str(uuid.uuid4())

    def get_all_workspaces(self):
        resp = requests.get(url=f"{self._base_url}/workspaces",
                            headers={
                                "Authorization": f"Bearer {self._token}"
                            },
                            verify=False).text
        return json.loads(resp)['data']

    def get_all_hosts_under_workspace(self, workspace_name):
        resp = requests.get(
            url=f"{self._base_url}/hosts?workspace={workspace_name}",
            headers={
                "Authorization": f"Bearer {self._token}"
            },
            verify=False).text
        return json.loads(resp)['data']

    def get_all_sessions_under_workspace(self, workspace_name):
        resp = requests.get(
            url=f"{self._base_url}/sessions?workspace={workspace_name}",
            headers={
                "Authorization": f"Bearer {self._token}"
            },
            verify=False).text
        return json.loads(resp)['data']

    def get_host_id_by_address(self, workspace_name, address):
        hosts = self.get_all_hosts_under_workspace(workspace_name)
        iters = filter(lambda item: item["address"] == address, hosts)
        id = next(iters)["id"]
        return id

    def get_sessions_binding_to_host(self, workspace_name, address):
        host_id = self.get_host_id_by_address(workspace_name, address)
        sessions = self.get_all_sessions_under_workspace(workspace_name)
        return list(filter(lambda item: item["host_id"] == host_id, sessions))

    def get_vuls_binding_to_host(self, host_id):
        resp = requests.get(f"{self._base_url}/vulns/{host_id}",
                            headers={
                                "Authorization": f"Bearer {self._token}"
                            },
                            verify=False).text
        print(resp)
        return json.loads(resp)['data']


def report_it(current_workspace, start_time, data):
    data['total_time'] = f"{time.time() - start_time}s"
    json_str = json.dumps(data)
    file_name = f'reports/{current_workspace}'
    with open(f'{file_name}.json', 'w') as jf:
        print(json_str, file=jf)
    visualize_the_report(f"{file_name}.json")
    print('report finishes')


def visualize_the_report(file_name):
    dot = Digraph('attack graph')
    with open(file_name, 'r') as f:
        json_bytes = f.read()
        report_result = json.loads(json_bytes)
        report_entities = report_result['report']
        dot.node("h-1",
                 f"attacker_host:{report_result['attacker_host']}",
                 shape="box")
        for r in report_entities:
            dot.node(
                f"h{r['order']}",
                f"host:{r['host']}",
                color="red",
                style='filled',
            )
        order_list = list(map(lambda a: a['order'], report_entities))
        for first, second in zip([-1] + order_list, order_list):
            dot.edge(
                f'h{first}',
                f'h{second}',
                label=
                f"cve:{report_entities[second]['cve']},\nexploit_via:{report_entities[second]['exploited_via']},\nexploit_module:{report_entities[second]['extra_info']['exploit_module'][0]}"
            )
        dot.render(f'{file_name}.dot', view=True)


r = ReportUtil(token=TOKEN)


def generate_report(nodes, sessions, current_workspace):
    result = []
    for index, node in enumerate(nodes):
        try:
            host_id = r.get_host_id_by_address(current_workspace, node._host)
            if host_id > 0:
                sessions = r.get_sessions_binding_to_host(
                    current_workspace, node._host)
                vuls_info = r.get_vuls_binding_to_host(host_id)
                result.append({
                    "order": index,
                    "host": node._host,
                    "host_up": True,
                    "exploited": True,
                    "exploited_via": "metasploit",
                    "cve": node._vul['cve'],
                    "extra_info": {
                        "exploit_module": [s["via_exploit"] for s in sessions],
                        "exploit_payload":
                        [s["via_payload"] for s in sessions],
                        "description":
                        " ".join([vuls_info['name'], vuls_info['info']]),
                        "refs":
                        list(map(lambda vul: vul['name'], vuls_info['refs']))
                    }
                })
            else:
                break
        except Exception as e:
            print(e)
    return {
        "workspace": current_workspace,
        "report": result,
        "attacker_host": get_current_host_ip()
    }
