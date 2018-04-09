#apriori ....FINAL

def loadDataSet():
    return [[1,2,5,6,10,13,15,16,17,18,20,21,23,25,26,38,41,42,43,44,45,49,54,56,57,58,59,62,63,64,65,66,67,70,72,73,75,77,79,81,82,83,84,89,92,93,94,95,96,97,99,101,102,106,108,109,117,120,121,124,125,128,130,131,134,137,138,141,144,145,148,150,151,157,158,160,162,168,174,177,178,181,182,184,189,193,194,198,199,200],
[1,5,13,22,30,42,49,64,72,83,87,92,95,102,110,130,178,193,197,200],
[1,43,49,59,62,63,81,82,95,99,104,130,145,157,160,181],
[1,7,10,12,13,16,18,19,22,43,49,59,62,64,77,83,84,87,92,94,99,102,109,115,130,144,151,158,160,189,194,197,198],
[1,13,21,28,43,44,72,92,102,109,118,130,135,145,188],
[1,9,18,63,71,76,79,90,181,198]]
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
              
    C1.sort()
    return list(map(frozenset, C1))#use frozen set so we
                            #can use it as a key in a dict   
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData
dataSet = loadDataSet()
print(dataSet)
C1 = createC1(dataSet)
#print(C1)    
#D is a dataset in the setform.

D = list(map(set,dataSet))
#print(D)
L1,suppDat0 = scanD(D,C1,0.9)
#print(L1)
def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList
def apriori(dataSet, minSupport = 0.9):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData
L,suppData = apriori(dataSet)
#print(L)
#print("single item frequent set")
#print(L[0])
#print("2 item frequent set")
#print(L[1])    
#print(L[2])
#print(L[3])
#print(aprioriGen(L[0],2) )
def generateRules(L, supportData, minConf=1.0):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList   
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
L,suppData= apriori(dataSet,minSupport=0.5)
rules= generateRules(L,suppData, minConf=0.7)
print(rules)

                   