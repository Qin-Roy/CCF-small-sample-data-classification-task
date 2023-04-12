import torch
from imblearn.over_sampling import SMOTE,ADASYN,KMeansSMOTE,BorderlineSMOTE,SVMSMOTE
from imblearn.combine import SMOTETomek,SMOTEENN

def smote(data):
    text_inputs = data['text_inputs']
    text_mask = data['text_mask']
    label = data['label']
    mix = torch.cat([text_inputs,text_mask],dim=1)
    label = label.squeeze(dim=1)
    smo = SMOTE(random_state=42, k_neighbors=2)
    mix_smo, label_smo = smo.fit_resample(mix, label)
    mix_smo = torch.LongTensor(mix_smo)
    label_smo = torch.LongTensor(label_smo)
    res_text_inputs,res_text_mask = mix_smo.split(len(text_inputs[0]),dim=1)
    data['text_inputs'] = res_text_inputs
    data['text_mask'] = res_text_mask
    data['label'] = label_smo.unsqueeze(dim=1)
    return data

def adasyn(data):
    text_inputs = data['text_inputs']
    text_mask = data['text_mask']
    label = data['label']
    mix = torch.cat([text_inputs,text_mask],dim=1)
    label = label.squeeze(dim=1)
    ada = ADASYN(random_state=42, n_neighbors=2)
    mix_smo, label_smo = ada.fit_resample(mix, label)
    mix_smo = torch.LongTensor(mix_smo)
    label_smo = torch.LongTensor(label_smo)
    res_text_inputs,res_text_mask = mix_smo.split(len(text_inputs[0]),dim=1)
    data['text_inputs'] = res_text_inputs
    data['text_mask'] = res_text_mask
    data['label'] = label_smo.unsqueeze(dim=1)
    return data

def blsmote(data):
    text_inputs = data['text_inputs']
    text_mask = data['text_mask']
    label = data['label']
    mix = torch.cat([text_inputs,text_mask],dim=1)
    label = label.squeeze(dim=1)
    kms = BorderlineSMOTE(random_state=42, k_neighbors=2)
    mix_smo, label_smo = kms.fit_resample(mix, label)
    mix_smo = torch.LongTensor(mix_smo)
    label_smo = torch.LongTensor(label_smo)
    res_text_inputs,res_text_mask = mix_smo.split(len(text_inputs[0]),dim=1)
    data['text_inputs'] = res_text_inputs
    data['text_mask'] = res_text_mask
    data['label'] = label_smo.unsqueeze(dim=1)
    return data

def kmsmote(data):
    text_inputs = data['text_inputs']
    text_mask = data['text_mask']
    label = data['label']
    mix = torch.cat([text_inputs,text_mask],dim=1)
    label = label.squeeze(dim=1)
    kms = KMeansSMOTE(random_state=42, k_neighbors=2)
    mix_smo, label_smo = kms.fit_resample(mix, label)
    mix_smo = torch.LongTensor(mix_smo)
    label_smo = torch.LongTensor(label_smo)
    res_text_inputs,res_text_mask = mix_smo.split(len(text_inputs[0]),dim=1)
    data['text_inputs'] = res_text_inputs
    data['text_mask'] = res_text_mask
    data['label'] = label_smo.unsqueeze(dim=1)
    return data

def svmsmote(data):
    text_inputs = data['text_inputs']
    text_mask = data['text_mask']
    label = data['label']
    mix = torch.cat([text_inputs,text_mask],dim=1)
    label = label.squeeze(dim=1)
    kms = SVMSMOTE(random_state=42, k_neighbors=2)
    mix_smo, label_smo = kms.fit_resample(mix, label)
    mix_smo = torch.LongTensor(mix_smo)
    label_smo = torch.LongTensor(label_smo)
    res_text_inputs,res_text_mask = mix_smo.split(len(text_inputs[0]),dim=1)
    data['text_inputs'] = res_text_inputs
    data['text_mask'] = res_text_mask
    data['label'] = label_smo.unsqueeze(dim=1)
    return data

def smote_tomek(data):
    text_inputs = data['text_inputs']
    text_mask = data['text_mask']
    label = data['label']
    mix = torch.cat([text_inputs,text_mask],dim=1)
    label = label.squeeze(dim=1)
    smo = SMOTETomek(random_state=42)
    mix_smo, label_smo = smo.fit_resample(mix, label)
    mix_smo = torch.LongTensor(mix_smo)
    label_smo = torch.LongTensor(label_smo)
    res_text_inputs,res_text_mask = mix_smo.split(len(text_inputs[0]),dim=1)
    data['text_inputs'] = res_text_inputs
    data['text_mask'] = res_text_mask
    data['label'] = label_smo.unsqueeze(dim=1)
    return data

def smote_enn(data):
    text_inputs = data['text_inputs']
    text_mask = data['text_mask']
    label = data['label']
    mix = torch.cat([text_inputs,text_mask],dim=1)
    label = label.squeeze(dim=1)
    smo = SMOTEENN(random_state=42)
    mix_smo, label_smo = smo.fit_resample(mix, label)
    mix_smo = torch.LongTensor(mix_smo)
    label_smo = torch.LongTensor(label_smo)
    res_text_inputs,res_text_mask = mix_smo.split(len(text_inputs[0]),dim=1)
    data['text_inputs'] = res_text_inputs
    data['text_mask'] = res_text_mask
    data['label'] = label_smo.unsqueeze(dim=1)
    return data

# test
# a=torch.rand(97,32)
# b=torch.rand(97,32)
# c=torch.cat([a,b],dim=1)
# print(c.size())
# d,e = c.split(32,dim=1)
# print(d.size())
# print(e.size())

# label=torch.tensor([[1],[2],[3]])
# label = label.squeeze(dim=1)
# print(label)
# label = label.unsqueeze(dim=1)
# print(label)
# text_inputs_all = torch.LongTensor(2,3)
# print(text_inputs_all)
# print(text_inputs_all[0].size())
# text_inputs_all[0]=torch.LongTensor([1,2,3])
# print(text_inputs_all)

