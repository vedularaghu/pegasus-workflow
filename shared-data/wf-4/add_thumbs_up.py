#! /usr/bin/env python3
import os, sys
import emoji

f1 = open(sys.argv[1])
lines = f1.readlines()

with open(sys.argv[2], 'w', encoding='utf-8-sig') as f:
    f.write(lines[0])
    f.write(emoji.emojize(':thumbs_up:'))
