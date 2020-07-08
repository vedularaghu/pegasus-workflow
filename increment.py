#!/usr/bin/env python3
import sys

f1 = open(sys.argv[1])

l1 = f1.readlines()
l = []
for i in l1:
    l.append(int(i.strip("\n"))+1)    

with open(sys.argv[2], "w") as f:
    for i in l:
        f.write(str(i)+"\n")

