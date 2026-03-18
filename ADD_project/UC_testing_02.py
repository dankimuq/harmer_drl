import harmat as hm
# if __name__ == "__main__":
# initialise the harm
from future.moves import sys
import networkx
from harmat.stats.analyse import *
from harmat import *
from retrieve_from_HARM import *
from harm_lang import *
import sys

def enterprise_network():
    Total_Num_node1 = 1
    enterprise = hm.Harm()

    # create the top layer of the harm
    enterprise.top_layer = hm.AttackGraph()

    A = hm.Attacker()  # attacker
    h8 = hm.Host("172.17.0.8")
    h2= hm.Host("172.17.0.2")
    h3 = hm.Host("172.17.0.3")
    h4 = hm.Host("172.17.0.4")
    h5 = hm.Host("172.17.0.5")
    h6 = hm.Host("172.17.0.6")
    h7 = hm.Host("172.17.0.7")


    # create some nodes
    # target


    #h1.portNumber = 'port:1080', 'port:80','port:53','port:6660'

    # then we will make a basic attack tree for host
    h8.lower_layer = hm.AttackTree()
    h2.lower_layer = hm.AttackTree()
    h3.lower_layer = hm.AttackTree()
    h4.lower_layer = hm.AttackTree()
    h5.lower_layer = hm.AttackTree()
    h6.lower_layer = hm.AttackTree()
    h7.lower_layer = hm.AttackTree()


    v1 = hm.Vulnerability("2011-3556",'port:80', values={'risk': 7.5, 'cost': 2.5, 'probability': 0.43, 'exploitability': 0.55,'impact': 5.5, 'defense_cost': 15})
    v2 = hm.Vulnerability("TCP timestamps",'port:1080', values={'risk': 2.6, 'cost': 7.9, 'probability': 0.21, 'exploitability': 0.29,'impact': 2.9, 'defense_cost': 15})
    v3 = hm.Vulnerability("DCE Services Enumeration Reporting",'port:6660',values={'risk': 10.0, 'cost': 1.0, 'probability': 0.43, 'exploitability': 0.55, 'impact': 5.5, 'defense_cost': 15})
    v4 = hm.Vulnerability("2013-0900",'port:53',values={'risk': 4.3, 'cost': 7.9, 'probability': 0.23, 'exploitability': 0.29,'impact': 2.9, 'defense_cost': 15})
    v5 = hm.Vulnerability("2012-1675",'port:8080',values={'risk': 7.5, 'cost': 3.7, 'probability': 0.43, 'exploitability': 0.55,'impact': 5.5, 'defense_cost': 18})
    v6 = hm.Vulnerability("2016-2834",'',values={'risk': 8.8, 'cost': 1.2, 'probability': 1.0, 'exploitability': 0.64,'impact': 6.4, 'defense_cost': 18})
    v7 = hm.Vulnerability("2017-0114",'port:8080',values={'risk': 9.3, 'cost': 1.2, 'probability': 0.43, 'exploitability': 0.64,'impact': 6.4, 'defense_cost': 18})




    h8.lower_layer.basic_at([v1])
    h2.lower_layer.basic_at([v1])
    h3.lower_layer.basic_at([v3])
    h4.lower_layer.basic_at([v1])
    h5.lower_layer.basic_at([v1, v2, v5])
    h6.lower_layer.basic_at([v6])
    h7.lower_layer.basic_at([v7])

    # add edges for servers
    enterprise[0].add_edge_between(A, h4)
    enterprise[0].add_edge_between(h4, h8)
    enterprise[0].add_edge_between(h4, h2)
    enterprise[0].add_edge_between(h4, h3)
    enterprise[0].add_edge_between(h4, h5)
    enterprise[0].add_edge_between(h4, h6)
    enterprise[0].add_edge_between(h4, h7)



    return enterprise

def simulation():
    harm = enterprise_network()
    Source = ""
    Target = ""
    for host in harm.top_layer.nodes():
        if host.name == "Attacker":
            Source = host
        if host.name == "172.17.0.8":
            Target = host
    harm[0].source = Source
    harm[0].target = Target
    harm[0].find_paths()
    harm.flowup()
    #hm.write_to_file(hm.convert_to_xml(harm), "C:\Python stuff\h\safeview\data\ADD_project\Zhibin_net.xml")
    #hm.HarmSummary(harm).show()


    #for path in retrieve_attack_paths(harm):
        #print path

    #for host, vuls in retrieve_path_info(harm).items():
        #for vul in vuls:
            #print host, vul
   # print 'Plan based on SP', retrieve_shortest_paths(harm)

    '''
    for host in harm[0].hosts():
        for vul in host.lower_layer.all_vulns():
            print vul.name, vul.service

    for host, edges in  retrieve_path_with_edges(harm).items():
        print host,':', edges
        
     '''


    print 'shortest path:', retrieve_shortest_paths(harm)


    print Get_enumerated_paths(harm)


    '''
    print 'information: hosts on the shortest paths............................'
    for tupleinfo in retrieve_path_info(harm).items():
        print tupleinfo
    
    print 'metric_based path', metric_based_attack_path(harm)

    for tupleinfo in retrieve_path_info_metric_based(harm).items():
        print tupleinfo
    '''

    #print 'metric_based path-',    metric_based_attack_path(harm) #incremental learning with target
    #for host in metric_based_attack_path(harm):
        #if host is not harm[0].source:
            #print host, host.cost
    #connections_attacker_knows(harm)

    #'''
    Network_host_connections(harm)
    print '________________________________________________________'

    for host in harm[0].hosts():
        if host.name=='172.17.0.8':
            Active_connection_from_host(harm, host)
    print '________________________________________________________'

    for host in harm[0].hosts():
        Attacker_knows(harm, host)
    print '________________________________________________________'


    #attack_plan(harm)
    print '________________________________________________________'
    #for node in harm[0].nodes():
        #print 'connection',node.name, ':', host_connections(harm, node)

    print '________________________________________________________'

    #for tupleinfo in retrieve_path_info(harm).items():
        #print 'serviceRunning', tupleinfo

    Get_AP_Host_info(harm)

    Result_metrics = open("C:\Output\\Attack_plan.txt", 'w')
    sys.stdout = Result_metrics

    #for path in networkx.all_simple_paths(harm[0], harm[0].source, harm[0].target):
        #print path

    #print retrieve_shortest_paths(harm) #shortest path
    print attack_plan(harm)

    #'''
    return


"""
------------------------------------------------------------------------------------------
Part: RUN SIMULATION
------------------------------------------------------------------------------------------
"""

simulation()
