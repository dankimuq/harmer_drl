#!/usr/bin/env python
#coding:utf-8
class VulNode:
    def __init__(self, host, vul_dict, **extras):
        "docstring"
        self._host = host
        self._vul = vul_dict
        self._extras = extras

    def __repr__(self):
        return f"hostname:{self._host}, vul:{self._vul}, extras:{self._extras}"

    def __str__(self):
        return self.__repr__()

    def whether_next(self):
        return True

    def __eq__(self, other):
        return (self._host == other._host) and (self._vul['cve']
                                                == other._vul['cve'])
