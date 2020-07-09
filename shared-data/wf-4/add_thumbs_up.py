import os, sys

f1 = open(sys.argv[1])
lines = f1.readlines()

print(lines)

with open(sys.argv[2], 'w', encoding='utf-8-sig') as f:
    f.write(lines[0])
    f.write('\U0001f44d')
