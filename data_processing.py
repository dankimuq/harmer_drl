#!/usr/bin/env python
import functools
import json

from node import VulNode


def parse_paths_file(filename):
    with open(filename, "r") as f:
        raw_paths_data = f.read()
    return eval(raw_paths_data)


def generate_path(raw_path):
    nodes = [
        dict(list(map(lambda item: item.split(':', maxsplit=1), raw_node)))
        for raw_node in raw_path
    ]
    return [
        VulNode(host=node['host'],
                vul_dict={
                    k: v
                    for k, v in node.items()
                    if ((not k == 'host') and (not k.startswith("ex")))
                },
                **{k: v
                   for k, v in node.items() if (k.startswith("ex"))})
        for node in nodes
    ]


def generate_paths(raw_paths):
    return [generate_path(raw_path) for raw_path in raw_paths]


def process_no_cve(paths):
    result = []
    for path in paths:
        index_of_no_cve = len(path)
        for i, n in enumerate(path):
            if "NOCVE" in str(n):
                index_of_no_cve = i
                break
        result.append(path[0:index_of_no_cve])
    result = list(filter(lambda data: data, result))
    result = eval(str(result).replace('Host', 'host').replace('CVE-', 'cve:'))
    return result


def transform_paths(paths):
    result = list(map(lambda x: list(map(lambda y: eval(y), x)), paths))
    return result


def compose(*functions):
    def compose2(f, g):
        return lambda x: f(g(x))

    return functools.reduce(compose2, functions, lambda x: x)


transform_function = compose(generate_paths, transform_paths, process_no_cve,
                             parse_paths_file)


def initialize_targets_config(filename):
    with open(filename, 'r') as f:
        targets_config = json.loads(f.read())
    return targets_config['approach'], {
        k: -1
        for k in targets_config['targets']
    }, targets_config['path-file']


def main():
    print(transform_function('Attack_plan_AttackCost.txt'))


if __name__ == "__main__":
    main()
