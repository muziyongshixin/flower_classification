import os
import random
all_path='/m/liyz/flower_classification/raw_data/all_filelist.txt'
train_path='/m/liyz/flower_classification/raw_data/train_filelist.txt'
dev_path='/m/liyz/flower_classification/raw_data/dev_filelist.txt'
test_path='/m/liyz/flower_classification/raw_data/test_filelist.txt'

all_sample=[]
train_sample=[]
dev_sample=[]

with open(all_path,"r") as f:
    for line in f.readlines():
        all_sample.append(line)

random.shuffle(all_sample)
train_sample=all_sample[0:3500]
dev_sample=all_sample[3500:]

with open(train_path,"w") as f:
    for line in train_sample:
        f.writelines(line)

with open(dev_path,"w") as f:
    for line in dev_sample:
        f.writelines(line)

with open(test_path,"w") as f :
    for i in range(424):
        info="/m/liyz/flower_classification/raw_data/test/%d.jpg -1"%i
        f.writelines(info+"\n")
