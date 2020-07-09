#!/usr/bin/env python3
import glob

nums = []
for f1 in glob.glob("nums*.txt"):
    f = open(f1)
    l1 = f.readlines()
    for i in l1:
        nums.append(int(i.strip("\n")))

odd = []
even = []
for i in nums:
    if i%2 == 0:
        even.append(i)
    else:
        odd.append(i)

with open("odd_nums.txt", "w") as f:
    for i in odd:
        f.write(str(i)+"\n")
with open("even_nums.txt", "w") as f:
    for i in even:
        f.write(str(i)+"\n")





