import torch
from torch.utils.data import SequentialSampler, DataLoader
import json
import numpy as np
from config import parse_args
from data_helper import MultiModalDataset
from model import clsModel
from util import *
from transformers import BertTokenizer

import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
def inference():
    args = parse_args()
    # 1. load data
    anns=list()
    labels = []
    with open(args.test_annotation,'r',encoding='utf8') as f:
        for line in f.readlines():
            ann =json.loads(line)
            labels.append(ann['label_id'])
            anns.append(ann)
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    test_list=[]
    for item in anns:
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
        test_list.append(data)
        
    dataset = MultiModalDataset(args, test_list)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)
    # 2. load model i
    models=[]
    for i in range(3):
        model = clsModel(args)
        save_path = f'save/flod_{i}'
        best_model = os.path.join(save_path, 'model_best.bin')
        checkpoint = torch.load(best_model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()
        models.append(model)
    # 3. inference
    all_outs=[]
    for model in models:
        print('infering')
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                outs = model(batch, inference=True,multi=True)
                predictions.extend(outs.cpu().numpy())
            predictions = np.array(predictions)
            all_outs.append(predictions)
    all_outs=np.array(all_outs)
    out = np.sum(all_outs,axis=0)
    predictions = np.argmax(out,axis=1)
    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        f.write(f'label_id,predicted_id\n')
        for pred_label_id, ann in zip(predictions, dataset.anns):
            label_id = ann['label']
            f.write(f'{label_id},{pred_label_id}\n')
    results = evaluate(predictions, labels)
    print(results)
if __name__ == '__main__':
    inference()