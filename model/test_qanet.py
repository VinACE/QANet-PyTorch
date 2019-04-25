import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.cnn import DepthwiseSeparableConv
from .modules.attention import MultiHeadAttention
from .modules.position import PositionalEncoding
from model.QANet import QANet


if __name__ == '__main__':
    model = QANet(
    wv_tensor,
    cv_tensor,
    args.para_limit,
    args.ques_limit,
    args.d_model,
    num_head=args.num_head,
    train_cemb=(not args.pretrained_char),
    pad=wv_word2ix["<PAD>"])
    model.summary()
    print(model.summary())