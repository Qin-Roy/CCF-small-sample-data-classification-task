import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from loss_func import *
# 直接 bert-base加载 roberta
# 最终特征采用 last 4 mean pooling，即取bert最后四层的特征平均池化
# 最终再接一个映射到36的分类头，即分36类
# loss部分即传统交叉熵，后续也可考虑focal loss 等

# # RDrop
# class clsModel(nn.Module):
#     def __init__(self, args):
#         super(clsModel, self).__init__()
#         self.bert = BertModel.from_pretrained(args.bert_dir, output_hidden_states=True)
#         # config = BertConfig(output_hidden_states=True)
#         # self.bert = BertModel(config=config)
#         self.cls = nn.Linear(768*4, 36)
#         self.text_embedding = self.bert.embeddings
#         self.text_cls = nn.Linear(768, 36)
#         self.dropout = nn.Dropout(p=args.dropout)
#         self.ce = nn.CrossEntropyLoss()
#         self.kld = nn.KLDivLoss(reduction="none")
#     def build_pre_input(self, data):
#         text_inputs=data['text_inputs']
#         text_mask=data['text_mask']
#         textembedding = self.text_embedding(text_inputs.cuda())
#         return textembedding,text_mask
#     def forward(self, data, inference=False,multi = False):
#         inputs_embeds, mask = self.build_pre_input(data)
#         bert_out = self.bert(attention_mask=mask, inputs_embeds=inputs_embeds)
#         # last 4 mean pooling
#         hidden_stats = bert_out.hidden_states[-4:] # 4*batch_size*sequence_length*768
#         hidden_stats = [i.mean(dim=1) for i in hidden_stats] # 4*batch_size*768
#         #torch.cat(hidden_stats,dim=1)   batch_size*(768*4)
#         out = self.cls(torch.cat(hidden_stats,dim=1)) # batch_size*36
#         out1 = self.dropout(out)
#         out2 = self.dropout(out)

#         if inference:
#             if multi:
#                 return out
#             else:
#                 return torch.argmax(out, dim=1)
#         else:
#             all_loss, all_acc, all_pre,label = self.cal_loss([out1,out2],data['label'].cuda())
#             return all_loss, all_acc, all_pre, label
#     def loss_fnc(self, y_pred, y_true, alpha=4):
#         """配合R-Drop的交叉熵损失
#             """
#         loss1 = self.ce(y_pred[0], y_true)
#         loss2 = self.kld(torch.log_softmax(y_pred[0], dim=1), y_pred[1].softmax(dim=-1)) + \
#                 self.kld(torch.log_softmax(y_pred[1], dim=1), y_pred[0].softmax(dim=-1))
#         return loss1 + torch.mean(loss2) / 4 * alpha
#     def cal_loss(self,prediction, label):
#         label = label.squeeze(dim=1)
#         loss = self.loss_fnc(y_pred=prediction, y_true=label)
#         with torch.no_grad():
#             pred_label_id = torch.argmax(prediction[1], dim=1)
#             accuracy = (label == pred_label_id).float().sum() / label.shape[0]
#         return loss, accuracy, pred_label_id, label

class clsModel(nn.Module):
    def __init__(self, args):
        super(clsModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, output_hidden_states=True)
        # config = BertConfig(output_hidden_states=True)
        # self.bert = BertModel(config=config)
        self.cls = nn.Linear(768*4, 36)
        self.text_embedding = self.bert.embeddings
        self.text_cls = nn.Linear(768, 36)
    def build_pre_input(self, data):
        text_inputs=data['text_inputs']
        text_mask=data['text_mask']
        textembedding = self.text_embedding(text_inputs.cuda())
        return textembedding,text_mask
    def forward(self, data, inference=False,multi = False):
        inputs_embeds, mask = self.build_pre_input(data)

        bert_out = self.bert(attention_mask=mask, inputs_embeds=inputs_embeds)
        # last 4 mean pooling
        hidden_stats = bert_out.hidden_states[-4:]
        hidden_stats = [i.mean(dim=1) for i in hidden_stats]
        out = self.cls(torch.cat(hidden_stats,dim=1))
        
        if inference:
            if multi:
                return out
            else:
                return torch.argmax(out, dim=1)
        else:
            all_loss, all_acc, all_pre,label = self.cal_loss(out,data['label'].cuda())
            return all_loss, all_acc, all_pre, label
    @staticmethod
    def cal_loss(prediction, label): 
        label = label.squeeze(dim=1)
        label_in = F.one_hot(label,num_classes=36).float()
    
        # loss = F.binary_cross_entropy_with_logits(prediction, label_in)
        # loss = F.cross_entropy(prediction, label)
        loss = F.cross_entropy(prediction, label_in)
        # loss = loss_func(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label