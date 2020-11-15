import numpy as np
import os
import pandas as pd
import sys
from six.moves import urllib
import numpy as np
from collections import defaultdict
from itertools import chain, combinations
from copy import deepcopy
from typing import Tuple
from collections import OrderedDict


import time

url1 = 'https://raw.githubusercontent.com/HInverse007/Frequent-Itemset-Mining-Project/main/BMS_dataset.txt'
url2 = 'https://raw.githubusercontent.com/HInverse007/Frequent-Itemset-Mining-Project/main/Bible_dataset.txt'
url3 = 'https://raw.githubusercontent.com/HInverse007/Frequent-Itemset-Mining-Project/main/Fifa_dataset.txt'
url4 = 'https://raw.githubusercontent.com/HInverse007/Frequent-Itemset-Mining-Project/main/Sign_dataset.txt'
url5 = 'https://raw.githubusercontent.com/Mudit-1999/Mining-Frequent-ItemSet/main/Bms_view2'
url6 = 'https://raw.githubusercontent.com/Mudit-1999/Mining-Frequent-ItemSet/main/Leviathan'
file = urllib.request.urlopen(url1)
dataset = list()
for line in file:
	decoded_line = line.decode("utf-8")
	temp = [ int(x) for x in decoded_line.split() if x.isdigit() ]
	temp.sort()
	dataset+=[temp]


def getItemSet(data_iterator,min_cnt):
  freqItem=list()
  itemSet = defaultdict(int)
  for record in data_iterator:
    for item in record:
      itemSet[item] +=1      
  for item, count in itemSet.items():
    if count >= min_cnt:
      freqItem+= [item]
  return freqItem,itemSet

class Node():
  def __init__(self, item=None, count=0):
      self.item = item
      self.count = count
      self.parent = None
      self.children = {}

class FPTree():
  def __init__(self):
      self.header_table = defaultdict(list)
      self.item_counter = defaultdict(int)
      self.root = Node()

  def add_path(self, tran, weight=1):
    cur = self.root
    for item in tran:
      if item in cur.children:
        cur.children[item].count += weight
        self.item_counter[item] += weight
        cur = cur.children[item]
        continue
      new_node = Node(item, weight)
      new_node.parent = cur
      cur.children[item] = new_node
      self.header_table[item]+=[new_node]
      self.item_counter[item] += weight
      cur = new_node
  def mine(self, min_cnt=1):
    # print("mine min_cnt: ",min_cnt)
    fp=list()
    fp_count =list()
    tmp = list(self.header_table)
    # print(tmp)
    tmp = sorted(tmp, key=lambda x: (self.item_counter[x], x), reverse=False) # the order is very important
    sorted_dict = OrderedDict([(el, self.header_table[el]) for el in tmp])
    # print(sorted_dict)
    self.header_table=sorted_dict


    for item in self.header_table:
      if self.item_counter[item] >= min_cnt:
        fp +=[[item]]
        fp_count+=[self.item_counter[item]]
        cond_trans, weights = self.get_conditional_tran(item, min_cnt)
        cond_tree = FPTree()
        for x in range(0,len(cond_trans)):
          cond_tree.add_path(cond_trans[x], weights[x])
        
        cond_fp, cond_fp_count = cond_tree.mine(min_cnt)
        # print(type(cond_fp),cond_fp)
        if cond_fp:
          local_cond_fp=list()
          for line in cond_fp:
            line+=[item]
            local_cond_fp += [line]
          fp += local_cond_fp
          fp_count += cond_fp_count

    if fp:
      sorted_fp=list()
      for x in fp:
        sorted_fp += [sorted(x)]
      tmp = list(zip(sorted_fp, fp_count))
      tmp = sorted(tmp, key=lambda x: (len(x[0]), x[0]))
      fp, fp_count = list(zip(*tmp))

    return fp, fp_count

  def get_conditional_tran(self, item, min_cnt=1):
    # print("cond min_cnt: ", min_cnt)
    transactions = list()
    weights = list()
    for node in self.header_table[item]:
      line = list()
      cur = node.parent
      while cur.item != None:
        line+=[cur.item]
        cur = cur.parent
      if line:
        transactions+=[line]
        weights+=[node.count]
    return transactions, weights



def fp_growth(dataSet, min_support=0.1):
    # print("fp min_support: ", min_support)
    # count and sort
    min_cnt = min_support * len(dataSet)
    # print("fp min_cnt: ",min_cnt)
    freqItem,itemSetCnt = getItemSet(dataSet,min_cnt)
    # print(freqItem,itemSetCnt)

    if len(freqItem)==0:
      return list()
    fp_tree = FPTree()

    # add trans
    for tran in dataSet:
      tmp=list()
      for item in tran:
        if item in freqItem:
          tmp+=[item]
      if len(tmp)==0:
        continue;
      tmp = list(set(tmp))
      tmp = sorted(tmp, key=lambda x: (itemSetCnt[x], x), reverse=True) # the order is very important
      fp_tree.add_path(tmp)
   
    # mine pattern
    res = fp_tree.mine(min_cnt)
    # fp_tree.print_tree()
    print("Exit")
    # for x,y in fp_tree.header_table.items():
      # print(x,fp_tree.item_counter[x])
    res = list(zip(*res))
    return res


class TopDownFPTree():
  def __init__(self):
      self.header_table = defaultdict(list)
      self.item_counter = defaultdict(int)
      self.root = Node()

  def add_path(self, tran, weight=1):
    cur = self.root
    for item in tran:
      if item in cur.children:
        cur.children[item].count += weight
        self.item_counter[item] += weight
        cur = cur.children[item]
        continue
      new_node = Node(item, weight)
      new_node.parent = cur
      cur.children[item] = new_node
      self.header_table[item]+=[new_node]
      self.item_counter[item] += weight
      cur = new_node

  def mine(self, min_cnt=1):
    # print("mine min_cnt: ",min_cnt)
    fp=list()
    fp_count =list()
    
    tmp = list(self.header_table)
    # print(tmp)
    tmp = sorted(tmp, key=lambda x: (self.item_counter[x], x), reverse=True) # the order is very important
    sorted_dict = OrderedDict([(el, self.header_table[el]) for el in tmp])
    # print(sorted_dict)
    self.header_table=sorted_dict


    for item in self.header_table:
      if self.item_counter[item] >= min_cnt:
        fp +=[[item]]
        fp_count+=[self.item_counter[item]]

        cond_trans, weights = self.get_top_down_conditional_tran(item, min_cnt)
        # print("Cond Tree",item)
        cond_tree = FPTree()
        for x in range(0,len(cond_trans)):
          cond_tree.add_path(cond_trans[x], weights[x])
        
        cond_fp, cond_fp_count = cond_tree.mine(min_cnt)
        # print("End Tree",item,cond_fp) 
        if cond_fp:
          local_cond_fp=list()
          for line in cond_fp:
            line=[item] + line
            local_cond_fp += [line]
          fp += local_cond_fp
          fp_count += cond_fp_count

    if fp:
      sorted_fp=list()
      for x in fp:
        sorted_fp += [sorted(x)]
      tmp = list(zip(sorted_fp, fp_count))
      tmp = sorted(tmp, key=lambda x: (len(x[0]), x[0]))
      fp, fp_count = list(zip(*tmp))

    return fp, fp_count


  def generate_all_trans(self,node,line_path,transactions,weights,level=0):
    if(len(node.children))==0:
      if node.count!=0 :
        transactions.append(line_path)
        weights+=[node.count]
      return node.count

    tmp_cnt=0
    for childItem,child in node.children.items():
      tmp_cnt+=self.generate_all_trans(child,line_path+[childItem],transactions,weights,level+1)
      
    if node.count > tmp_cnt and len(line_path)!=0:
      transactions.append(line_path)
      weights+=[(node.count-tmp_cnt)]

    return node.count


  def get_top_down_conditional_tran(self, item, min_cnt=1):
    transactions = list()
    weights = list()
    for node in self.header_table[item]:
      self.generate_all_trans(node,list(),transactions,weights)
    return transactions, weights


def td_fp_growth(dataSet, min_support=0.1):
    min_cnt = min_support * len(dataSet)
    # print("td_fp min_cnt: ",min_cnt)
    freqItem,itemSetCnt = getItemSet(dataSet,min_cnt)
    if len(freqItem)==0:
      return list()
    td_fp_tree = TopDownFPTree()

    # add trans
    for tran in dataSet:
      tmp=list()
      for item in tran:
        if item in freqItem:
          tmp+=[item]
      if len(tmp)==0:
        continue;
      tmp = list(set(tmp))
      tmp = sorted(tmp, key=lambda x: (itemSetCnt[x], x), reverse=True) # the order is very important
      td_fp_tree.add_path(tmp)
   
    res = td_fp_tree.mine(min_cnt)
    res = list(zip(*res))
    return res

# Bottom Up Implementation

start_time=time.time()
min_support = 0.05
fps = fp_growth(dataset, min_support)
end_time=time.time()
print("Bottom Up Approach")
print("Min Support:",min_support)
print("Number of FI:",len(fps))
print("Time for running Simple Apriori Algorithm is",(end_time - start_time),"sec")

# Top Down Implementation

start_time=time.time()
min_support = 0.05
fps = td_fp_growth(dataset, min_support)
end_time=time.time()
print("Top Down Approach")
print("Min Support:",min_support)
print("Number of FI:",len(fps))
print("Time for running Top Down Apriori Algorithm is",(end_time - start_time),"sec")
