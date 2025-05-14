import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            # A = F.softmax(A, dim=1)  # softmax over N
            A = A.sigmoid()
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
            # A = A.sigmoid()

        return A  ### K x N

from Models.causal import CausalModel
class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes,causal,v_dim,cat=False):
        super(Classifier_1fc, self).__init__()
        self.causal=causal
        if causal:
            self.causal = CausalModel(q_dim=512, k_dim=n_channels, v_dim=v_dim,cat=cat).cuda()

        if cat:
            self.fc = nn.Linear(v_dim*2, n_classes)
        else:
            self.fc = nn.Linear(v_dim, n_classes)


    def forward(self, data):
        x=data[0]

        if self.causal:
            cluster=data[1]
            x=self.causal.forward(x,cluster)

        pred = self.fc(x)

        return pred, x, None


class Attention_with_Classifier(nn.Module):
    def __init__(self, in_size, L=512, D=128, K=1, num_cls=1):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)

        self.classifier = Classifier_1fc(in_size, num_cls,False,in_size)

    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        M = torch.mm(AA, x) ## K x L
        pred, _, _ = self.classifier(M) ## K x num_cls
        return pred, M, AA


