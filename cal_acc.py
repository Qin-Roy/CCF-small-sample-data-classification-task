import codecs
import csv

import jsonlines
import json

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

#count_train_label_num
label_id_num=[0]*36

f_in = open("./data/train_new36.json", "r", encoding="utf8")
for item in jsonlines.Reader(f_in):
    label_id_num[item['label_id']]+=1
print(label_id_num)

f_out = open("./csv_acc/smoke_double_lr.csv", 'w')
f_out.write(f'label_id,sum,right,acc,train_num\n')

data=[[] for i in range(36)]
with codecs.open('./csv/smoke_double_lr.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f, skipinitialspace=True):
        data[int(row['label_id'])].append(row['predicted_id'])
# print(data)
count = 0
for i in range(0,36):
    sum=len(data[i])
    for item in data[i]:
        if int(item) == i:
            count+=1
    acc=round(float(count*100)/sum,2)
    print(str(i)+",sum:"+str(sum)+",true:"+str(count)+",acc:"+str(acc))
    f_out.write(f'{i},{sum},{count},{acc},{label_id_num[i]}\n')
    count=0

