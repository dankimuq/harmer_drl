#!/usr/bin/env python3
# coding:utf-8

import json
import os
import signal
import sys
import time
from argparse import ArgumentParser
from functools import partial
from pprint import pprint

from basic_usage import BasicUsage
from configuration import (APPROACH_NAME, PATH_FILE, START_TIMESTAMP, TARGETS,
                           TOKEN)
from data_processing import transform_function
from py_client import client
from report import generate_report, r, report_it
from traverser import nodes_exploited, tranverse_the_nodes
from vul_searcher import search_vul_by_keywords


def main():
    parser = ArgumentParser(description='Execution Mode')
    parser.add_argument('--advance',
                        type=bool,
                        help='Advance Mode',
                        default=False,
                        required=False)
    args = parser.parse_args()
    bu = BasicUsage(client)
    current_workspace = bu.generate_new_workspace()

    paths_of_nodes = transform_function(PATH_FILE)

    for i, path_of_nodes in enumerate(paths_of_nodes):
        print(f"current index {i}")
        tranverse_the_nodes(
            path_of_nodes,
            bu,
            current_workspace,
            advance_mode=args.advance,
            is_top_level_func=True,
        )
    print(f'total time: {time.time()-START_TIMESTAMP} s')
    with open(f'reports/{APPROACH_NAME}.txt', 'w') as f:
        f.write(json.dumps(TARGETS))
    print('All finished')
    report_it(
        current_workspace, START_TIMESTAMP,
        generate_report(nodes_exploited,
                        r.get_all_sessions_under_workspace(current_workspace),
                        current_workspace))
    os.kill(os.getpid(), signal.SIGKILL)


if __name__ == '__main__':
    main()
