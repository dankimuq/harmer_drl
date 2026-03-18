#!/usr/bin/env python

from py_client import client


def search_attack_module_by_vul_keyword(keywords, cb=None):
    if 'type' not in keywords:
        keywords['type'] = 'exploit'
    search_str = " ".join([f"{k}:{v}" for k, v in keywords.items()])
    resp = client.call('module.search', [search_str])
    return [d['fullname'] for d in resp]


def main():
    result = search_attack_module_by_vul_keyword({
        'cve': '2011-3556',
        'type': 'exploit'
    })
    print(result)


if __name__ == '__main__':
    main()
