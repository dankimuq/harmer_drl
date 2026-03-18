import itertools



print list(itertools.permutations(['D1','D2','D3'], 3))
#for item in list(itertools.permutations(['D1','D2','D3'], 3)):
    #print item

#returned as a generator. Use list(permutations(l)) to return as a list.)
#print list(itertools.permutations([1,2,3,4], 2))