import harmat as hm
# if __name__ == "__main__":
# initialise the harm
from future.moves import sys
import networkx
from harmat.stats.analyse import *
from harmat import *
from retrieve_from_HARM import *

#Access attack paths
def retrieve_attack_paths(network):
    list_paths=[]
    for path in networkx.all_simple_paths(network[0], network[0].source, network[0].target):
        list_paths.append(path)
    return list_paths


def host_connections (network, host_name):
    'returns the connections w.r.t. one particular host (e.g., host 1)'
    list_attacker_knows = network[0].edges([host_name])
    list_con = []
    for edge in list_attacker_knows:
        #print list_attacker_knows
        list_con.append(edge[1])

    return list_con


def Network_host_connections(network):
    'Returns all connections (set) w.r.t. each host in the network'
    for node in network[0].hosts():
        #
        for con_node in host_connections(network, node):
            #print 'connected', '(',node.name,',', con_node.name,')', 'requires', 'host','(',node.name,')',',','host','(',con_node.name,')'
            print 'connected', '(', node.name, ',', con_node.name, ')'


def Active_connection_from_host(network, host_name):
    'returns a '
    list_con=[]
    #for connection in host_connections (network, host_name):
        #list_con.append(connection[1])
    #print 'hostVisible', host_name.name,':', list_con
    #print 'knowsConnected', host_name.name, ':', host_connections (network, host_name)
    list_node = host_connections(network, host_name)
    new_list=[]
    for node in list_node:
        new_list.append(node.name)
    print 'hostVisible', new_list


def Attacker_knows (network, host_name):
    list_attacker_knows = network[0].edges([host_name])
    print 'knowsConnected', list_attacker_knows


def all_shortest_paths(network):
    leng = len(network[0].shortest_path_length())
    sp= []
    for path in retrieve_attack_paths(harm): #list of paths
        if len(path)== leng:
            sp.append(path)
       # print path

    path_risk_sum = 0
    for path in sp:
        for node in path[1:]:
            if hasattr(node, 'impact'):
                node_risk = node.impact * node.probability
            else:
                node_risk = node.risk
            path_risk_sum += node_risk
        print (path, path_risk_sum)

def retrieve_shortest_paths(network):
    'ATTACK Plan - Get the shortest path with the max risk value'
    list_sp = []
    for path in networkx.all_simple_paths(network[0], network[0].source, network[0].target):
        if len(path[1:]) == network[0].shortest_path_length():
            list_sp.append(path)
    sp_path_with_max_risk = ''
    max_risk = 0.0
    for path in list_sp:
        path_risk = network[0].path_risk(path)
        if path_risk > max_risk:
            max_risk = path_risk
            sp_path_with_max_risk = path
    return sp_path_with_max_risk[1:] #while skipiing the attacker

def attack_plan(network):
    list_path=[]

    for host in retrieve_shortest_paths(network):
        host_vul = []
        for vul in host.lower_layer.all_vulns():
            host_vul.append('host:'+host.name)
            host_vul.append('cve:'+vul.name)
        list_path.append(host_vul)
    return list_path


def retrieve_path_info(network):
    'Get Information about hosts on the shortest paths'
    dict_host_on_path_info={}

    for node in retrieve_shortest_paths(network):
        if node.name != 'Attacker':
            #get all the vuls on a host
            list_of_vulns = []
            vulns = [(vul.service, vul.name, vul.risk) for vul in node.lower_layer.all_vulns()]
            list_of_vulns.extend(vulns)

            #dict_host_on_path_info[node.name] =vulns, network[0].edges([node])
            dict_host_on_path_info[node.name] = vulns
    return dict_host_on_path_info

def Get_AP_Host_info(network):
    for tupleinfo in retrieve_path_info(network).items():
        print 'Host_has', tupleinfo



def Get_enumerated_paths(network):
    'Get Information about hosts on the shortest paths to make an attack path'
    list_host_on_path_info=[]

    for node in retrieve_shortest_paths(network):

        if node.name != 'Attacker':
            path = []
            for vul in node.lower_layer.all_vulns():
                path.append('host:'+node.name)
                path.append('cve:'+vul.name)


            #for v in list_of_vulns:
                #dict_host_on_path_info['host:'+node.name] = 'cve:'+v
            list_host_on_path_info.append(path)
    return list_host_on_path_info