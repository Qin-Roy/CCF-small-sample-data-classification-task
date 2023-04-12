import jsonlines
import json
import re

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

# 遍历读取json对象
f2 = open('./data/train_balanced_split.json', 'w', encoding="utf8")
f = open("./data/train_balanced.json", "r", encoding="utf8")
count = 0
for item in jsonlines.Reader(f):
    # print(item)
    item['statement'] = ''
    item['programme'] = ''
    item['effect'] = ''
    text = item['abstract']
    pattern = r'。'
    test_text = item['abstract']
    result_list = re.split(pattern, test_text)
    result_list=list(filter(None, result_list))
    # print(len(result_list))
    if len(result_list) >= 3:
        for i in range(len(result_list)):
            if i == 0:
                item['statement'] = result_list[i]
            elif i < len(result_list)-1:
                item['programme'] += result_list[i]
            elif i == len(result_list)-1:
                item['effect'] = result_list[i]
    else:
        pattern = r'。|；'
        result_list = re.split(pattern, test_text)
        result_list=list(filter(None, result_list))
        if len(result_list) >= 3:
            for i in range(len(result_list)):
                if i == 0:
                    item['statement'] = result_list[i]
                elif i < len(result_list)-1:
                    item['programme'] += result_list[i]
                elif i == len(result_list)-1:
                    item['effect'] = result_list[i]
        else:
            pattern = r'。|；|，'
            result_list = re.split(pattern, test_text)
            result_list=list(filter(None, result_list))
            if len(result_list) >= 3:
                for i in range(len(result_list)):
                    if i < len(result_list)//3:
                        item['statement'] += result_list[i]
                    elif len(result_list)//3 <= i < len(result_list)//3 * 2:
                        item['programme'] += result_list[i]
                    else:
                        item['effect'] += result_list[i]
            else:
                pattern = r'。|；|，|、'
                result_list = re.split(pattern, test_text)
                result_list=list(filter(None, result_list))
                if len(result_list) < 3:
                    # print(item['id']+"："+str(len(result_list)))
                    if len(result_list) == 1:
                        item['statement'] = result_list[0]
                        item['programme'] = result_list[0]
                        item['effect'] = result_list[0]
                    elif len(result_list) == 2:
                        item['statement'] = result_list[0]
                        item['programme'] = result_list[1]
                        item['effect'] = result_list[1]
                else:
                    for i in range(len(result_list)):
                        if i < len(result_list)//3:
                            item['statement'] += result_list[i]
                        elif len(result_list)//3 <= i < len(result_list)//3 * 2:
                            item['programme'] += result_list[i]
                        else:
                            item['effect'] += result_list[i]

    del item['abstract']
    json.dump(item, f2, ensure_ascii=False)
    f2.write("\n")



