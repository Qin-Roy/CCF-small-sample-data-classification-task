# import jsonlines
# import json

# import io
# import sys
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

# # label_id_num=[33,19,183,33,44,36,47,48,39,25,52,54,7,16,19,33,17,13,16,13,25,12,5,22,29,16,17,22,8,7,13,4,5,8,12,6]
# label_id_num=[37, 21, 192, 29, 42, 36, 49, 56, 36, 24, 53, 54, 8, 17, 19, 35, 14, 15, 16, 15, 25, 14, 4, 23, 32, 17, 14, 20, 8, 8, 14, 4, 4, 8, 10, 6]

# f_out = open("./data/train_new36_balanced.json", "w", encoding="utf8")
# f_in = open("./data/train_new36.json", "r", encoding="utf8")
# per_num = 200

# for item in jsonlines.Reader(f_in):
#     repeat_time = per_num // label_id_num[item['label_id']]
#     if repeat_time < 1:
#         repeat_time += 1
#     # print(str(item['label_id'])+":"+str(repeat_time))
#     for i in range(repeat_time):
#         json.dump(item, f_out, ensure_ascii=False)
#         f_out.write("\n")


import codecs
import csv

import jsonlines
import json

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
# label_id_num=[33,19,183,33,44,36,47,48,39,25,52,54,7,16,19,33,17,13,16,13,25,12,5,22,29,16,17,22,8,7,13,4,5,8,12,6]
label_id_num=[37, 21, 192, 29, 42, 36, 49, 56, 36, 24, 53, 54, 8, 17, 19, 35, 14, 15, 16, 15, 25, 14, 4, 23, 32, 17, 14, 20, 8, 8, 14, 4, 4, 8, 10, 6]
f_out = open("./distribute.csv", 'w')
f_out.write(f'label_id,train_num\n')


for i in range(0,36):
    f_out.write(f'{i},{label_id_num[i]}\n')



