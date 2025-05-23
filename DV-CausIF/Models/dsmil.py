import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#vpt
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from functools import reduce
from operator import mul
from Models.causal import CausalModel

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size,cluster_dim,causal, v_dim,cat=False,dropout_v=0.0, nonlinear=0, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        self.input_size = input_size
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)

        self.causal=causal
        if causal:
            self.causal_model = CausalModel(q_dim=v_dim, k_dim=cluster_dim, v_dim=v_dim,cat=cat).cuda()
        if cat:
            self.fcc = nn.Conv1d(1, 1, kernel_size=v_dim * 2)
        else:
            self.fcc = nn.Conv1d(1, 1, kernel_size=v_dim )

    def forward(self, feats, c,cluster): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        # print(m_indices.shape)
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
        # B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        # C = self.fcc(B).view(1, -1)

        if self.causal:
            B = self.causal_model.forward(B,cluster)
            # C=C+M
        C = self.fcc(B).view(1, -1)

        return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self, in_size,out_size,i_classifier, b_classifier,causal):
        super(MILNet, self).__init__()
        self.projecter = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU()
        )
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.causal=causal
    def forward(self, data):
        x=data[0]
        x=self.projecter(x)
        feats, classes = self.i_classifier(x)
        # print(feats)
        if self.causal:
            cluster=data[1]
        else:
            cluster=None
        prediction_bag, A, B = self.b_classifier(feats, classes,cluster)
        
        return classes, prediction_bag, A
        