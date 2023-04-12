from util_loss import ResampleLoss,MultiDSCLoss,GHMC_Loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class_freq = [37, 21, 192, 29, 42, 36, 49, 56, 36, 24, 53, 54, 8, 17, 19, 35, 14, 15, 16, 15, 25, 14, 4, 23, 32, 17, 14, 20, 8, 8, 14, 4, 4, 8, 10, 6]
train_num = 979

# #Diceloss
# loss_func = MultiDSCLoss(alpha=1.0, smooth=1.0, reduction="mean")

# # GHM
# loss_func = GHMC_Loss(bins=10,alpha=0.5)

# #BCE
# loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
#                              focal=dict(focal=False, alpha=0.5, gamma=2),
#                              logit_reg=dict(),
#                              class_freq=class_freq, train_num=train_num)

# #FL
# loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
#                              focal=dict(focal=True, alpha=0.5, gamma=2),
#                              logit_reg=dict(),
#                              class_freq=class_freq, train_num=train_num)
    
# #CB
# loss_func = ResampleLoss(reweight_func='CB', loss_weight=10.0,
#                              focal=dict(focal=True, alpha=0.5, gamma=2),
#                              logit_reg=dict(),
#                              CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
#                              class_freq=class_freq, train_num=train_num) 
    
# #R-BCE-Focal
# loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0, 
#                              focal=dict(focal=True, alpha=0.5, gamma=2),
#                              logit_reg=dict(),
#                              map_param=dict(alpha=0.1, beta=10.0, gamma=0.9), 
#                              class_freq=class_freq, train_num=train_num)

# #NTR-Focal
# loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
#                              focal=dict(focal=True, alpha=0.5, gamma=2),
#                              logit_reg=dict(init_bias=0.05, neg_scale=2.0),
#                              class_freq=class_freq, train_num=train_num)
    
# #DBloss-noFocal
# loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=0.5,
#                              focal=dict(focal=False, alpha=0.5, gamma=2),
#                              logit_reg=dict(init_bias=0.05, neg_scale=2.0),
#                              map_param=dict(alpha=0.1, beta=10.0, gamma=0.9), 
#                              class_freq=class_freq, train_num=train_num)

# #CBloss-ntr
# loss_func = ResampleLoss(reweight_func='CB', loss_weight=10.0,
#                              focal=dict(focal=True, alpha=0.5, gamma=2),
#                              logit_reg=dict(init_bias=0.05, neg_scale=2.0),
#                              CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
#                              class_freq=class_freq, train_num=train_num)
    
# DB
loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.9), 
                             class_freq=class_freq, train_num=train_num)

