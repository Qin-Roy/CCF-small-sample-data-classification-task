import torch
import numpy as np

def mixup_data(batch, time=1):
    length = batch['text_inputs'].shape[0]
    for _ in range(length * time):
        a = np.random.randint(0, length - 1)
        b = np.random.randint(0, length - 1)
        lam = np.random.random()
        mixup_text_input = (lam * batch['text_inputs'][a] + (1 - lam) * batch['text_inputs'][b]).view(1, 494).long()
        mixup_text_mask = (lam * batch['text_mask'][a] + (1 - lam) * batch['text_mask'][b]).view(1, 494).long()
        # mixup_text_type_ids = (lam * batch['text_type_ids'][a] + (1 - lam) * batch['text_type_ids'][b]).view(1, 494).long()
        mixup_label = (lam * batch['label'][a] + (1 - lam) * batch['label'][b]).view(1, 1).long()
        # print(mixup_text_input)
        # print(mixup_text_mask)
        # print(mixup_text_type_ids)
        # print(mixup_label)
        # assert False, 'manual error'
        batch['text_inputs'] = torch.cat((batch['text_inputs'], mixup_text_input))
        batch['text_mask'] = torch.cat((batch['text_mask'], mixup_text_mask))
        # batch['text_type_ids'] = torch.cat((batch['text_type_ids'], mixup_text_type_ids))
        batch['label'] = torch.cat((batch['label'], mixup_label))
    return batch