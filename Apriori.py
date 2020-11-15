import numpy as np
import os
import pandas as pd
import sys
from six.moves import urllib
import numpy as np
from collections import defaultdict
from itertools import chain, combinations
import time
import pandas as pd 

dataset = []
NoOfTransactions=0
minSupport =0.02
k_break=100
partitionSize=3
isprint=True
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  
def getItemSetAndtnsList(dataSet):
    tnsList = list()
    itemSet = set()
    for record in dataSet:
      # print(record)
      transaction = frozenset(record)
      tnsList.append(transaction)
      for item in transaction:
        itemSet.add(frozenset([item]))         
    return itemSet, tnsList

def returnItemsWithMinSupport(itemSet, tnsList, minSupport, freqSet):
  _itemSet = set()
  localSet = defaultdict(int)

  for transaction in tnsList:
    for item in itemSet:
      if item.issubset(transaction):
        localSet[item] += 1

  for item, count in localSet.items():
    if float(count) >= minSupport*len(tnsList):
      freqSet[item] = count
      _itemSet.add(item)

  return _itemSet

def returnItemsWithMinSupportAndtnsList(itemSet, tnsList, minSupport, freqSet, NoOfTransactions):
  _itemSet = set()
  _tnsList=list()
  localSet = defaultdict(int)

  for transaction in tnsList:
    flag=False
    for item in itemSet:
      if item.issubset(transaction):
        localSet[item] += 1
        flag=True
    if flag:
      _tnsList.append(transaction)

  for item, count in localSet.items():
    if float(count) >= minSupport*NoOfTransactions:
      freqSet[item] = count
      _itemSet.add(item)

  return _itemSet,_tnsList

def joinSet(itemSet, length):
  skip=0
  res = [sorted(list(lis)[0:length]) for lis in itemSet] 
  res.sort()
  small_res=[]
  finalSet=[]
  x=0
  while x< len(res):
    skip=1
    tmp=res[x][0:length-2]
    small_res=set()
    small_res.add(frozenset(res[x][0:length-1]))
    for j in range(x+1,len(res)):
      if res[j][0:length-2] != tmp:
        break
      else:
        skip+=1
        small_res.add(frozenset(res[j]))
    x+=skip
    # print(small_res)
    finalSet= finalSet+ [i.union(j) for i in small_res for j in small_res if len(i.union(j)) == length]
  return set(finalSet)


def pruneSet(itemSet,prevSet,k):
  _itemSet =set()
  for i in itemSet:
    subsets = frozenset(combinations(i, k-1))
    flag=True
    for j in subsets:
      if frozenset(j) not in prevSet:
        flag=False
        break
    if flag:
      _itemSet.add(i)
  return _itemSet

def SimpleApriori(dataSet,isprint=True,isBreak=False):
    FIM=set()
    freqSet = defaultdict(int)
    start_time=time.time()
    itemSet, tnsList = getItemSetAndtnsList(dataSet)
    currentSet = returnItemsWithMinSupport(itemSet,tnsList,minSupport,freqSet)
    FIM|=currentSet
    k = 2
    while(currentSet != set([])):
        prevSet    = currentSet
        currentSet = joinSet(currentSet, k)
        currentSet = pruneSet(currentSet,prevSet,k)
        currentSet = returnItemsWithMinSupport(currentSet,tnsList,minSupport,freqSet)
        FIM|=currentSet
        if k==k_break and isBreak:
            break
        k = k + 1
    end_time=time.time()
    if isprint:
        print("Simple Apriori")
        print("Min Support:",minSupport)
        if isBreak:
            print("Lenght of MaxFequent Itemset")
        print("Number of FI:",len(FIM))
        print("Time for running Simple Apriori Algorithm is",(end_time - start_time),"sec")
    return FIM


def Partioning(dataSet,isprint=True):
  # Pass 1
  max_time=0
  if isprint:
    print("Partitioning")
    print("Min Support:",minSupport)
  n_split = np.array_split(dataSet, partitionSize)
  largeFreqSet = set()
  for t,partition in enumerate(n_split):
    start_time=time.time()
    largeFreqSet|=SimpleApriori(partition,False)
    end_time=time.time()
    print("Time for patition",t, " is " ,(end_time - start_time),"sec")
    max_time=max(max_time,(end_time - start_time))
  
  # Pass 2
  freqSet = defaultdict(int)
  start_time=time.time()
  _, tnsList = getItemSetAndtnsList(dataSet)
  currentSet = returnItemsWithMinSupport(largeFreqSet,tnsList,minSupport,freqSet)
  end_time=time.time()
  if isprint:
    print("Number of FI:",len(currentSet))
    print("Time for second pass for partioning algorithm is" ,(end_time - start_time),"sec")
  return currentSet


def transaction_reduction(dataSet,isprint=True,isBreak=False):
    if isprint:
        print("Transaction Reduction")
    FIM=set()
    freqSet = defaultdict(int)
    start_time=time.time()
    itemSet, tnsList = getItemSetAndtnsList(dataSet)
    currentSet,tnsList = returnItemsWithMinSupportAndtnsList(itemSet,tnsList,minSupport,freqSet,NoOfTransactions)
    print(len(tnsList))
    FIM|=currentSet
    k = 2
    while(currentSet != set([])):
        prevSet    = currentSet
        currentSet = joinSet(currentSet, k)
        currentSet = pruneSet(currentSet,prevSet,k)
        currentSet,tnsList = returnItemsWithMinSupportAndtnsList(currentSet,tnsList,minSupport,freqSet,NoOfTransactions)
        FIM|=currentSet
        if k==k_break and isBreak:
            break
        k = k + 1
    end_time=time.time()
    if isprint:
        print("Min Support:",minSupport)
        if isBreak:
            print("Lenght of MaxFequent Itemset")
        print("Number of FI:",len(FIM))
        print("Time for running Simple Apriori Algorithm is",(end_time - start_time),"sec")
    return FIM

if __name__ == "__main__":
    print("Press 1 for BMS Web View1")
    print("Press 2 for Bible")
    print("Press 3 for BMS Web View2")
    datasetNumber= int(input())
    urlFile=None
    if datasetNumber ==1:
        urlFile = 'https://raw.githubusercontent.com/HInverse007/Frequent-Itemset-Mining-Project/main/BMS_dataset.txt'
    elif  datasetNumber ==2:
        urlFile = 'https://raw.githubusercontent.com/HInverse007/Frequent-Itemset-Mining-Project/main/Bible_dataset.txt'
    elif  datasetNumber ==3:
        urlFile = 'https://raw.githubusercontent.com/Mudit-1999/Mining-Frequent-ItemSet/main/Bms_view2'
    else:
        print("Invalid input")
        exit(0)
    file = urllib.request.urlopen(urlFile)
    for line in file:
        decoded_line = line.decode("utf-8")
        temp = [ int(x) for x in decoded_line.split() if x.isdigit() ]
        temp.sort()
        dataset.append(temp)
    NoOfTransactions=len(dataset)
    minSupport=float(input("MinSupport between 0 and 1: "))
    partitionSize=int(input("Enter partitionsize for Partioning algorithm: "))
    kprune=input("Enter Yes for setting max FI length: ")
    if kprune=="Yes":
        k_break=int(input("Max FI Length: "))
        SimpleApriori(dataset,isBreak=True)
    SimpleApriori(dataset)
    Partioning(dataset)
    transaction_reduction(dataset)
