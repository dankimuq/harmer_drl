import harmat as hm
# if __name__ == "__main__":
# initialise the harm
from future.moves import sys
import networkx
from harmat.stats.analyse import *
from harmat import *

def enterprise_network():
    Total_Num_node1 = 1
    enterprise = hm.Harm()

    # create the top layer of the harm
    enterprise.top_layer = hm.AttackGraph()

    A = hm.Attacker()  # attacker
    h1 = hm.Host("h1")
    h2= hm.Host("h2")
    h3 = hm.Host("h3")
    h4 = hm.Host("h4")
    h5 = hm.Host("h5")
    # create some nodes
    # target


    # then we will make a basic attack tree for host
    h1.lower_layer = hm.AttackTree()
    h2.lower_layer = hm.AttackTree()
    h3.lower_layer = hm.AttackTree()
    h4.lower_layer = hm.AttackTree()
    h5.lower_layer = hm.AttackTree()

    # WS1
    alpha = 0.9
    v1 = hm.Vulnerability("CVE-2015-3185", values={'risk': 4.3, 'cost': 5.7, 'probability': 0.43, 'exploitability': 0.55,'impact': 5.5, 'defense_cost': 15})
    v2 = hm.Vulnerability("CVE-2015-5700", values={'risk': 2.1, 'cost': 7.9, 'probability': 0.21, 'exploitability': 0.29,'impact': 2.9, 'defense_cost': 15})

    # WS2
    v3 = hm.Vulnerability("CVE-2015-3185",values={'risk': 4.3, 'cost': 5.7, 'probability': 0.43, 'exploitability': 0.55, 'impact': 5.5, 'defense_cost': 15})
    v4 = hm.Vulnerability("CVE-2015-5700",values={'risk': 2.1, 'cost': 7.9, 'probability': 0.23, 'exploitability': 0.29,'impact': 2.9, 'defense_cost': 15})

    # AS1
    v5 = hm.Vulnerability("CVE-2015-0900",values={'risk': 4.3, 'cost': 5.7, 'probability': 0.43, 'exploitability': 0.55,'impact': 5.5, 'defense_cost': 18})
    v6 = hm.Vulnerability("CVE-2013-0638",values={'risk': 10.0, 'cost': 0.1, 'probability': 1.0, 'exploitability': 0.64,'impact': 6.4, 'defense_cost': 18})

    # AS2
    v7 = hm.Vulnerability("CVE-2016-0763",values={'risk': 4.3, 'cost': 5.7, 'probability': 0.43, 'exploitability': 0.64,'impact': 6.4, 'defense_cost': 18})
    v8 = hm.Vulnerability("CVE-2015-0900",values={'risk': 4.3, 'cost': 5.7, 'probability': 0.43, 'exploitability': 0.55,'impact': 5.5, 'defense_cost': 18})
    # DB3
    v9 = hm.Vulnerability("CVE-2012-1675",values={'risk': 7.5, 'cost': 2.5, 'probability': 0.75, 'exploitability': 0.64,'impact': 6.4, 'defense_cost': 20})
    v10 = hm.Vulnerability("CVE-2015-0900",values={'risk': 4.3, 'cost': 5.7, 'probability': 0.43, 'exploitability': 0.55,'impact': 5.5, 'defense_cost': 20})
    # add vulnerabilities to host nodes
    h1.lower_layer.basic_at([v1, v2])
    h2.lower_layer.basic_at([v3, v4])
    h3.lower_layer.basic_at([v5, v6])
    h4.lower_layer.basic_at([v7, v8])
    h5.lower_layer.basic_at([v9, v10])
    # add edges for servers
    enterprise[0].add_edge_between(A, [h1, h2])
    enterprise[0].add_edge_between(h1, [h3, h2])
    enterprise[0].add_edge_between(h2, [h3, h4])
    enterprise[0].add_edge_between(h3, h4)
    enterprise[0].add_edge_between(h3, h5)
    enterprise[0].add_edge_between(h4, h5)
    return enterprise

def simulation():
    harm = enterprise_network()
    Source = ""
    Target = ""
    for host in harm.top_layer.nodes():
        if host.name == "Attacker":
            Source = host
        if host.name == "h5":
            Target = host
    harm[0].source = Source
    harm[0].target = Target
    harm[0].find_paths()
    harm.flowup()
    #hm.write_to_file(hm.convert_to_xml(tharm), "C:\Python stuff\h\safeview\data\Simon_simulations\example_net.xml")
    #hm.HarmSummary(harm).show()
    sp = harm[0].shortest_path_length()
    #print harm.risk
    #print harm.cost
    #print sp
    #print harm[0].number_of_attack_paths()

    list_sp = []
    #print 'paths...'
    for path in networkx.all_simple_paths(harm[0], harm[0].source, harm[0].target):

        #print (path)
        if len(path[1:])== sp:
            #print(path)
            list_sp.append(path)
    print 'shortest paths...'
    #print list_sp
    #for p in list_sp:
        #print p
    sp_path_with_max_risk = ''
    max_risk = 0.0
    for path in list_sp:
        r = harm[0].path_risk(path)
        #print r, path
        if r>max_risk:
            max_risk=r
            sp_path_with_max_risk=path
    print '>>>>>', sp_path_with_max_risk

    return


"""
------------------------------------------------------------------------------------------
Part: RUN SIMULATION
------------------------------------------------------------------------------------------
"""

simulation()
