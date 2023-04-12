import json
import random
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer
import random
from sklearn.model_selection import  train_test_split,StratifiedKFold
from mixup import mixup_data
from smote import smote
def create_dataloaders(args, test_mode = False):
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    val_ratio = args.val_ratio
    anns=list()
    with open(args.train_annotation,'r',encoding='utf8') as f:
        for line in f.readlines():
            ann =json.loads(line)
            anns.append(ann)
    random.shuffle(anns)
    val_anns = anns[:int(val_ratio*len(anns))]
    train_anns = anns[int(val_ratio*len(anns)):]
    train_anns = train_anns + train_anns
    val_list=[]
    train_list=[]
    for item in val_anns:
        sentence = item['sentence']
        text_inputs = {}
        sentence_inputs = tokenizer(sentence, max_length= 495, padding='max_length', truncation=True)
        sentence_inputs['input_ids'][0] = 101
        sentence_inputs['input_ids'] = sentence_inputs['input_ids'][1:]
        sentence_inputs['attention_mask'] = sentence_inputs['attention_mask'][1:]
        sentence_inputs['token_type_ids'] = sentence_inputs['token_type_ids'][1:] 
        for each in sentence_inputs:
            text_inputs[each] = sentence_inputs[each]
        text_inputs = {k: torch.LongTensor(v) for k,v in text_inputs.items()}
        text_mask = text_inputs['attention_mask']
        data = dict(
            text_inputs=text_inputs['input_ids'],
            text_mask=text_mask,
            text_type_ids = text_inputs['token_type_ids'],
        )
        data['label'] = torch.LongTensor([item['label_id']])
        val_list.append(data)
        

    for item in train_anns:
        sentence = item['sentence']
        text_inputs = {}
        sentence_inputs = tokenizer(sentence, max_length= 495, padding='max_length', truncation=True)
        sentence_inputs['input_ids'][0] = 101
        sentence_inputs['input_ids'] = sentence_inputs['input_ids'][1:]
        sentence_inputs['attention_mask'] = sentence_inputs['attention_mask'][1:]
        sentence_inputs['token_type_ids'] = sentence_inputs['token_type_ids'][1:] 
        for each in sentence_inputs:
            text_inputs[each] = sentence_inputs[each]
        text_inputs = {k: torch.LongTensor(v) for k,v in text_inputs.items()}
        text_mask = text_inputs['attention_mask']
        data = dict(
            text_inputs=text_inputs['input_ids'],
            text_mask=text_mask,
            text_type_ids = text_inputs['token_type_ids'],
        )
        data['label'] = torch.LongTensor([item['label_id']])
        train_list.append(data)
    
    train_len = len(train_list)
    input_id_len = len(train_list[0]['text_inputs'])
    text_inputs_all = torch.LongTensor(train_len,input_id_len)
    text_mask_all = torch.LongTensor(train_len,input_id_len)
    labels_all = torch.LongTensor(train_len,1)
    for i in range(0,train_len):
        text_inputs_all[i] = train_list[i]['text_inputs']
        text_mask_all[i] = train_list[i]['text_mask']
        labels_all[i] = train_list[i]['label']
    data_expand = dict(
        text_inputs = text_inputs_all,
        text_mask = text_mask_all,
        label = labels_all,
    )
    data_expand = mixup_data(data_expand)
    data_expand = smote(data_expand)
    # data_expand = adasyn(data_expand)
    # data_expand = blsmote(data_expand)
    # data_expand = kmsmote(data_expand)
    # data_expand = svmsmote(data_expand)
    # data_expand = tomek_links(data_expand)
    # data_expand = smote_enn(data_expand)
    # data_expand = smote_enn(data_expand)

    train_list=[]
    new_train_len = len(data_expand['text_inputs'])
    for i in range(0,new_train_len):
        item = dict(
            text_inputs = data_expand['text_inputs'][i],
            text_mask = data_expand['text_mask'][i],
            label = data_expand['label'][i],
        )
        train_list.append(item)
    
    # repeat <offline enhance>
    # train_anns = train_anns + train_anns
    val_dataset = MultiModalDataset(args, val_list)
    train_dataset = MultiModalDataset(args, train_list)
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                sampler=val_sampler,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataloader, val_dataloader
class MultiModalDataset(Dataset):
    def __init__(self,
                 args,
                 anns,
                 test_mode: bool = False,
                 idx= [] ):
        self.test_mode = test_mode
        if test_mode:
            self.tokenizer = BertTokenizer.from_pretrained(args.test_bert_dir)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        self.anns=anns
    def __len__(self) -> int:
        return len(self.anns)

    def __getitem__(self, idx: int) -> dict:
        return self.anns[idx]