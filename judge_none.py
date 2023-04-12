import jsonlines
import json
import re

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

# 遍历读取json对象
f = open("./data/train_split.json", "r", encoding="utf8")
for item in jsonlines.Reader(f):
    if item['statement'] == '':
        print("statement:"+item['id'])
    if item['programme'] == '':
        print("programme:"+item['id'])
    if item['effect'] == '':
        print("effect:"+item['id'])
   


