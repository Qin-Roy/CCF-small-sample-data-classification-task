import jsonlines
import json

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

# label_id_num=[33,19,183,33,44,36,47,48,39,25,52,54,7,16,19,33,17,13,16,13,25,12,5,22,29,16,17,22,8,7,13,4,5,8,12,6]
# f_in = open("./data/train.json", "r", encoding="utf8")
# per_num=200

# count=0
# label=-1
# for item in jsonlines.Reader(f_in):
#     if item['label_id'] != label:
#         if label != -1:
#             print("label_id:"+str(label)+":"+str(count))
#         count = 1
#         label = item['label_id']
#     else:
#         count+=1
#         #print(count)
# print("label_id:"+"35"+":"+str(count))


# count_label_id
# label_id_num=[0]*36

# f_in = open("./data/train_new36.json", "r", encoding="utf8")
# for item in jsonlines.Reader(f_in):
#     label_id_num[item['label_id']]+=1
# print(label_id_num)
# for i in range(0,36):
#     print(str(i)+":"+str(label_id_num[i]))

# #生成新的36类
# f_out = open("./data_new/train_new36.json", "w", encoding="utf8")
# f_in = open("./data_new/train.json", "r", encoding="utf8")
# label_id_num=[[] for i in range(119)]
# label_id_new=[4,11,17,8,16,19,28,21,25,13,20,36,33,26,82,22,35,45,59,85,94,101,104,91,113,12,58,99,92,93,102,105,112,84,75,87]
# for item in jsonlines.Reader(f_in):
#     label_id_num[int(item['label'])].append(item)
# # for i in range(0,119):
# #     print(str(i)+":"+str(len(label_id_num[i])))
# for i in range(0,36):
#     for item in label_id_num[label_id_new[i]]:
#         item['label_id'] = i
#         del item['label']
#         json.dump(item, f_out, ensure_ascii=False)
#         f_out.write("\n")


