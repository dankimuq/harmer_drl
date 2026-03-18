import harmat as hm
# if __name__ == "__main__":
# initialise the harm
from future.moves import sys
import networkx
from harmat.stats.analyse import *
from harmat import *
def next_max_risk_node(network, node):
    #entry_point = network[0].find_node("Attacker")
    lis_of_neigbors = network[0].neighbors(node)

    new_list_of_neigbors=[]
    for node in lis_of_neigbors:
        if node.name != 'Attacker':
            new_list_of_neigbors.append(node)
    #print 'NN', new_list_of_neigbors

    node_with_max_risk = ''
    max_risk = 0.0
    for node in new_list_of_neigbors:
        node_risk = node.risk
        if node_risk > max_risk:
            max_risk = node_risk
            node_with_max_risk = node
    return node_with_max_risk

def metric_based_attack_path(network):
    paths = []
    #entry_point = network[0].find_node("Attacker")
    entry_point=network[0].source
    paths.append(entry_point)

    Target = network[0].target
    Source = network[0].source

    while paths[-1] is not Target:
        #print paths[-1]
        the_node = next_max_risk_node(network, paths[-1])
        paths.append(the_node)

        if paths[-1] is Target:
            break

    return paths

def retrieve_path_info_metric_based(network):
    dict_host_on_path_info={}
    for node in metric_based_attack_path(network):
        if node.name != 'Attacker':
            #get all the vuls on a host
            list_of_vulns = []
            vulns = [(vul.service,vul.name, vul.risk) for vul in node.lower_layer.all_vulns()]
            list_of_vulns.extend(vulns)
            dict_host_on_path_info[node.name] =vulns, network[0].edges([node])
            #dict_host_on_path_info[node.name] = vulns
    return dict_host_on_path_info



