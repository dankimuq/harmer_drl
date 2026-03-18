import harmat as hm
# if __name__ == "__main__":
# initialise the harm
from future.moves import sys
import networkx
from harmat.stats.analyse import *
from harmat import *
import random
from retrieve_from_HARM import *

def load_field_test_harm():
    harm = hm.parse_xml("C:\Python stuff\h\safeview\data\ADD_project\UC_field_harm.xml") #reading the field harm from the folder
    'connection'
    #'''
    density = 0.4
    for host1 in harm[0]:
        for host2 in harm[0]:
            if random.random() < density and host1 is not host2 :
                harm[0].add_edge_between(host1, host2)
    #'''
    Target = ""
    for host in harm.top_layer.nodes():
        if host.name == "132.181.147.232":
            Target = host

    harm[0].target = Target

    harm.top_layer.remove_node(harm.top_layer.find_node("132.181.147.254"))  # switch

    harm[0].find_paths()

    #for path in networkx.all_simple_paths(harm[0], harm[0].source, harm[0].target):
        #print (path)



    #for node in harm[0].hosts():
        #if node.name !='Attacker':
        #for vul in node.lower_layer.all_vulns():
            #print node.name, vul.name
    print '............................'

    #print 'shortest path:', retrieve_shortest_paths(harm)
    #print 'information: hosts on the shortest paths............................'
    #for tupleinfo in retrieve_path_info(harm).items():
        #print tupleinfo

    print '............................'
    print 'metric-based path', metric_based_attack_path(harm)

    for tupleinfo in retrieve_path_info_metric_based(harm).items():
        print tupleinfo


    return

load_field_test_harm()



































































def load_field_test_harm222():
    harm = hm.parse_xml("C:\Python stuff\h\safeview\data\ADD_project\UC_field_harm.xml")
    #harm.top_layer.remove_node(harm.top_layer.find_node("Attacker"))
    #harm.top_layer.remove_node(harm.top_layer.find_node("132.181.147.254")) #switch
    #A = hm.Attacker()  # attacker
    'connection'
    density=0.4
    for host1 in harm[0]:
        for host2 in harm[0]:
            if random.random() < density and host1 is not host2:
                harm[0].add_edge_between(host1, host2)
    Source = ""
    Target = ""
    entry_point = ""
    for host in harm.top_layer.nodes():
        if host.name == "A":
            Source = host
        if host.name == "132.181.147.232":
            Target = host
        if host.name == "132.181.147.227": #Entry point node
            entry_point = host
    harm[0].source = Source
    harm[0].target = Target
    #harm.top_layer.add_edge_between(A, [entry_point])
    hm.write_to_file(hm.convert_to_xml(harm), "C:\Python stuff\h\safeview\data\ADD_project\harm.xml")
    harm[0].find_paths()
    #harm.flowup()
    #for tuple in retrieve_path_info(harm).items():
        #print tuple

    return


