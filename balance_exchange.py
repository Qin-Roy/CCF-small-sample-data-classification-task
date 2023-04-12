import jsonlines
import json
import random
import string

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

label_id_num=[33,19,183,33,44,36,47,48,39,25,52,54,7,16,19,33,17,13,16,13,25,12,5,22,29,16,17,22,8,7,13,4,5,8,12,6]
label_id_list=[0]
count=0;
for i in range(len(label_id_num)):
    count+=label_id_num[i]
    label_id_list.append(count)
print(label_id_list)
# [0, 33, 52, 235, 268, 312, 348, 395, 443, 482, 507, 559, 613, 620, 636, 655, 688, 705, 718, 
# 734, 747, 772, 784, 789, 811, 840, 856, 873, 895, 903, 910, 923, 927, 932, 940, 952, 958]

f_out = open("./data/train_balanced_exchanged.json", "w", encoding="utf8")
f_in = open("./data/train.json", "r", encoding="utf8")
per_num = 200

info=[]

for item in jsonlines.Reader(f_in):
    info.append(item)
#print(len(info))

for i in range(len(label_id_num)):
    #print(str(label_id_list[i])+"-"+str(label_id_list[i+1]))
    for j in range(label_id_list[i],label_id_list[i+1]):
        json.dump(info[j], f_out, ensure_ascii=False)
        f_out.write("\n")
    count=label_id_num[i]
    while count < 200:
        title_index = random.randrange(label_id_list[i], label_id_list[i+1])
        assignee_index = random.randrange(label_id_list[i], label_id_list[i+1])
        abstract_index = random.randrange(label_id_list[i], label_id_list[i+1])
        print(str(count+1)+":"+str(title_index)+"-"+str(assignee_index)+"-"+str(abstract_index))
        item['id']=info[title_index]['id']
        item['title']=info[title_index]['title']
        item['assignee']=info[assignee_index]['assignee']
        item['abstract']=info[abstract_index]['abstract']
        item['label_id']=info[abstract_index]['label_id']
        json.dump(item, f_out, ensure_ascii=False)
        f_out.write("\n")
        count+=1

    

    


