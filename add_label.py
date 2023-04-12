import jsonlines
import json

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

# # 遍历读取json对象
# with open('./data/testC.json', 'w', encoding="utf8") as f2:
#     with open("./data/testB.json", "r", encoding="utf-8") as f:
#         for item in jsonlines.Reader(f):
#             print(item)
#             item['label_id'] = 0
#             json.dump(item, f2, ensure_ascii=False)
#             f2.write("\n")

# 将label换为label_id
with open('./data_new/dev_new.json', 'w', encoding="utf8") as f2:
    with open("./data_new/dev.json", "r", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            item['label_id'] = int(item['label'])
            del item['label']
            json.dump(item, f2, ensure_ascii=False)
            f2.write("\n")

# # 添加将label换为label_id
# with open('./data_new/train_new.json', 'w', encoding="utf8") as f2:
#     with open("./data_new/train.json", "r", encoding="utf-8") as f:
#         for item in jsonlines.Reader(f):
#             item['label_id'] = 0
#             json.dump(item, f2, ensure_ascii=False)
#             f2.write("\n")