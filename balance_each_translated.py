import jsonlines
import json

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

# label_id_num=[33,19,183,33,44,36,47,48,39,25,52,54,7,16,19,33,17,13,16,13,25,12,5,22,29,16,17,22,8,7,13,4,5,8,12,6]
label_id_num=[37, 21, 192, 29, 42, 36, 49, 56, 36, 24, 53, 54, 8, 17, 19, 35, 14, 15, 16, 15, 25, 14, 4, 23, 32, 17, 14, 20, 8, 8, 14, 4, 4, 8, 10, 6]

f_out = open("./data/train_new36_balanced_translated.json", "w", encoding="utf8")
f_in1 = open("./data/train_new36.json", "r", encoding="utf8")
f_in2 = open("./data/train_new36_translated.json", "r", encoding="utf8")

info1=[]
info2=[]

for item in jsonlines.Reader(f_in1):
    info1.append(item)
#print(len(info1))
for item in jsonlines.Reader(f_in2):
    info2.append(item)
#print(len(info2))

per_num = 200
for i in range(len(info1)):
    repeat_time = per_num // label_id_num[info1[i]['label_id']]
    if repeat_time < 1:
        repeat_time += 1
    # print(str(item['label_id'])+":"+str(repeat_time))
    for j in range(0,repeat_time):
        if j%2==0:
            json.dump(info1[i], f_out, ensure_ascii=False)
        else:
            json.dump(info2[i], f_out, ensure_ascii=False)
        f_out.write("\n")


