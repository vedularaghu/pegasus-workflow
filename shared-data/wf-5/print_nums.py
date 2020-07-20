#!/usr/bin/env python3
import signal
import time
import sys

f = open("./saved_stated.txt")
n = int(f.readline().rstrip())

def final_append(signal, frame):
    with open("./saved_stated.txt", "w") as f1:
        f1.write(str(n))

signal.signal(signal.SIGTERM, final_append)

for i in range(n, 181):
    n += 1
    print(n)
    time.sleep(1)
    