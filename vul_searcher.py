import re
import sys
import time
from threading import Thread

from pymetasploit3.msfconsole import MsfRpcConsole

from basic_usage import BasicUsage
from py_client import client

console = MsfRpcConsole(client)

GLOABL_RESULT = {}


def search_vul_by_keywords(keywords, cb=None):
    search_str = "search"
    if 'type' not in keywords:
        keywords['type'] = 'exploit'
    for k, v in keywords.items():
        search_str = f"{search_str} {k}:{v}"
    print(search_str)
    console.callback = cb
    console.execute(f"grep exploit {search_str}")


def validate_search_result(search_result):
    print(f'validate:{search_result}')
    if "[]" == search_result:
        print('will failed')
        return False

    return True


def main():
    def extract(data):

        splited_data = data['data'].split()
        exploits = [
            splited_data[i + 1].split("/", maxsplit=1)[1]
            for i, d in enumerate(splited_data) if re.match(r'^[0-9]+$', d)
        ]
        #[1].split("/", maxsplit=1)[1]
        return exploits

    search_vul_by_keywords({'cve': '2011-3556', 'type': 'exploit'}, cb=extract)


if __name__ == '__main__':
    main()
