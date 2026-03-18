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
from itertools import combinations
import itertools
import copy

def enterprise_network():
    Total_Num_node1 = 1
    enterprise = hm.Harm()

    # create the top layer of the harm
    enterprise.top_layer = hm.AttackGraph()

    A = hm.Attacker()  # attacker
    h1 = hm.Host("206.171.47.1")
    h2= hm.Host("206.171.47.2")
    h3 = hm.Host("206.171.47.3")
    h4 = hm.Host("206.171.47.4")
    h5 = hm.Host("206.171.47.5") #set as target
    h6 = hm.Host("206.171.47.6")
    h7 = hm.Host("206.171.47.7")


    # create some nodes
    # target


    #h1.portNumber = 'port:1080', 'port:80','port:53','port:6660'

    # then we will make a basic attack tree for host
    h1.lower_layer = hm.AttackTree()
    h2.lower_layer = hm.AttackTree()
    h3.lower_layer = hm.AttackTree()
    h4.lower_layer = hm.AttackTree()
    h5.lower_layer = hm.AttackTree()
    h6.lower_layer = hm.AttackTree()
    h7.lower_layer = hm.AttackTree()


    v1 = hm.Vulnerability("DCE Services Enumeration Reporting",'port:135', values={'risk': 5.0, 'cost': 5.0, 'probability': 0.50, 'exploitability': 0.5,'impact': 5., 'defense_cost': 15})
    v2 = hm.Vulnerability("SMBv1 Unspecified Remote Code Execution (Shadow Brokers)",'port:445', values={'risk': 10.0, 'cost': 1.0, 'probability': 1.0, 'exploitability': 0.1,'impact': 9.0, 'defense_cost': 15})
    v3 = hm.Vulnerability("TCP timestamps",'port:general',values={'risk': 2.6, 'cost': 7.4, 'probability': 0.26, 'exploitability': 0.26, 'impact': 3, 'defense_cost': 15})
    v4 = hm.Vulnerability("SSL/TLS: Certificate Signed Using A Weak Signature Algorithm",'port:5443',values={'risk': 4.0, 'cost': 6.0, 'probability': 0.40, 'exploitability': 0.6,'impact': 5, 'defense_cost': 15})
    v5 = hm.Vulnerability("SSL/TLS: Diffie-Hellman Key Exchange Insufficient DH Group Strength Vulnerability",'port:443',values={'risk': 4.0, 'cost': 6.0, 'probability': 0.4, 'exploitability': 0.6,'impact': 5.5, 'defense_cost': 18})
    v6 = hm.Vulnerability("SSL/TLS:Report Weak Cipher Suites",'3389',values={'risk': 4.3, 'cost': 6.7, 'probability': 0.43, 'exploitability': 0.67,'impact': 6.4, 'defense_cost': 18})


    h1.lower_layer.basic_at([v2, v3])
    h2.lower_layer.basic_at([v3])
    h3.lower_layer.basic_at([v1, v2, v3])
    h4.lower_layer.basic_at([v1, v2, v3, v4,v5, v6])
    h5.lower_layer.basic_at([v1, v2, v3, v4,v5, v6])
    h6.lower_layer.basic_at([v3])
    h7.lower_layer.basic_at([v1, v2, v3, v4,v5, v6])

    # add edges for servers

    enterprise[0].add_edge_between(A, h1)
    enterprise[0].add_edge_between(A, h2)
    enterprise[0].add_edge_between(h1, h3)
    enterprise[0].add_edge_between(h1, h4)
    enterprise[0].add_edge_between(h1, h7)
    enterprise[0].add_edge_between(h2, h3)
    enterprise[0].add_edge_between(h3, h5)
    enterprise[0].add_edge_between(h3, h7)
    enterprise[0].add_edge_between(h5, h7)
    enterprise[0].add_edge_between(h4, h6)
    enterprise[0].add_edge_between(h6, h7)

    return enterprise


def rSubset(arr, r):
    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r)))

    return list(combinations(arr, r))


def defense(set_defense, harm):
    print set_defense
    for defense in set_defense:
        if defense =='D1':
            for host in harm[0].hosts():
                if host.name=='206.171.47.1':
                    for vul in host.lower_layer.all_vulns():
                        if vul.name =="SMBv1 Unspecified Remote Code Execution (Shadow Brokers)":
                            host.lower_layer.patch_vul(vul)
            harm[0].find_paths()
            harm.flowup()
            print 'risk:', harm.risk, ',', 'NAP:', harm[0].number_of_attack_paths()



        if defense =='D2':
            for host in harm[0].hosts():
                if host.name=='206.171.47.7':
                    for vul in host.lower_layer.all_vulns():
                        if vul.name =="SMBv1 Unspecified Remote Code Execution (Shadow Brokers)":
                            host.lower_layer.patch_vul(vul)
            harm[0].find_paths()
            harm.flowup()
            print 'risk:', harm.risk, ',', 'NAP:', harm[0].number_of_attack_paths()


        if defense =='D3':
            for host in harm[0].hosts():
                if host.name=='206.171.47.2':
                    for vul in host.lower_layer.all_vulns():
                        if vul.name =="TCP timestamps":
                            host.lower_layer.patch_vul(vul)
            harm[0].find_paths()
            harm.flowup()
            print 'risk:', harm.risk, ',', 'NAP:', harm[0].number_of_attack_paths()

    print '.......................'










def simulation():
    harm = enterprise_network()
    Source = ""
    Target = ""
    for host in harm.top_layer.nodes():
        if host.name == "Attacker":
            Source = host
        if host.name == "206.171.47.7":
            Target = host
    harm[0].source = Source
    harm[0].target = Target
    harm[0].find_paths()
    harm.flowup()

    #hm.write_to_file(hm.convert_to_xml(harm), "C:\Python stuff\h\safeview\data\ADD_project\Framework.xml")
    #hm.HarmSummary(harm).show()


    for path in retrieve_attack_paths(harm):
        print path



    print 'before defense:', 'risk:', harm.risk, ',', 'NAP:', harm[0].number_of_attack_paths()

    set_defense = ['D1', 'D2', 'D3']
    list_combn = list(itertools.permutations(set_defense, 3))
    for def_combn in list_combn:

        harm_copy = copy.deepcopy(harm)
        defense(def_combn, harm_copy)



    return


"""
------------------------------------------------------------------------------------------
Part: RUN SIMULATION
------------------------------------------------------------------------------------------
"""

simulation()
