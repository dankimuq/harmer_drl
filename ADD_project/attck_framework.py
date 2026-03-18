from pyattck import Attck
attack = Attck()
for actor in attack.actors:
    #print actor.name
    if 'APT1' in actor.name:
        if actor.name=='APT1':
            print actor.name
            #print actor

        for technique in actor.techniques:
            print(technique.name)
    #print actor


        for malware in actor.malwares:
            print(malware.name)
            # accessing techniques that this malware is used in



# accessing malware
#for malware in attack.malwares:
    #print(malware.name)
    #if malware.name == 'WEBC2':

        # accessing techniques that this malware is used in
        #for technique in malware.techniques:
            #print(technique._tactic)




# accessing techniques
#for technique in attack.techniques:
    #print(technique.name)
    #print('................................')

    # accessing tactics that this technique belongs to
    #for tactic in technique.tactics:
        #print(tactic.name)

    # accessing mitigation recommendations for this technique
    #for mitigation in technique.mitigations:
        #print(mitigation)

    # accessing actors using this technique
    #for actor in technique.actors:
        #print(actor)
