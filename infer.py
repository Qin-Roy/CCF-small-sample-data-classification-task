import torch
from torch.utils.data import SequentialSampler, DataLoader
import os
import json
from config import parse_args
from model import clsModel
from tqdm import tqdm 
from data_helper import MultiModalDataset
from util import *
from transformers import BertTokenizer

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
def inference():
    args = parse_args()
    print(args.ckpt_file)
    print(args.test_batch_size)
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
    # 2. load model
    model = clsModel(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    new_key = model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    # model.half()
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()
    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pred_label_id = model(data = batch,inference=True)
            predictions.extend(pred_label_id.cpu().numpy())
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