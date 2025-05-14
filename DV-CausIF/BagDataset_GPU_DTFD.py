import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
import torch


class KMeansGPU:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers = None

    def fit(self, X):
        # 初始化聚类中心
        initial_indices = torch.randperm(X.size(0))[:self.n_clusters]
        self.cluster_centers = X[initial_indices].clone()

        for iteration in range(self.max_iter):
            # 计算每个样本到聚类中心的距离
            distances = torch.cdist(X, self.cluster_centers)

            # 分配簇
            labels = torch.argmin(distances, dim=1)

            # 计算新的聚类中心
            new_centers = torch.stack([X[labels == k].mean(dim=0) for k in range(self.n_clusters)])

            # 检查收敛
            center_shift = torch.norm(new_centers - self.cluster_centers)
            if center_shift < self.tol:
                break

            # 更新聚类中心
            self.cluster_centers = new_centers

        return self.cluster_centers



class BagDataset_gpu(Dataset):
    def __init__(self, data,split_path, sheet_name,args) -> None:
        super(BagDataset_gpu).__init__()
        self.args = args
        file_labels_df = pd.read_excel(split_path, sheet_name=sheet_name)
        train_slide_names = file_labels_df['file_name'].tolist()
        pt_labels = file_labels_df['label'].tolist()


        x ,topk= [],[]
        for i,slide_name in enumerate(train_slide_names):
            bag = torch.load(data + "/" + slide_name + ".pt", map_location="cuda")
            if len(bag)<10:
                continue

            x.append(bag)
        self.x = x

        y=[]
        for pt_label in pt_labels:
            label = torch.tensor([pt_label],dtype=torch.float).to("cuda")
            y.append(label)
        self.y=y
        self.global_centroids=0

    def build_cluster(self,dimReduction,DTFDattention,n_clusters,per):

        dimReduction.eval()
        DTFDattention.eval()

        topk=[]
        for i,bag in enumerate(self.x):
            bag_num=int(len(bag)*per)
            with torch.no_grad():
                tmidFeat = dimReduction(bag)
                attention = DTFDattention(tmidFeat).squeeze()

                attention=attention.detach()
                if len(attention.size())>1:
                    attention=attention.squeeze()
                sort_attn,index=attention.sort()
                sample_bag=bag[index[:bag_num]]
                topk.append(sample_bag)
        centroids = torch.cat(topk, dim=0)

        kmeans = KMeansGPU(n_clusters=n_clusters)
        global_centroids=kmeans.fit(centroids)


        if self.global_centroids ==0:
            return 0.9*self.global_centroids+0.1*global_centroids
        else:
            self.global_centroids=global_centroids
            return global_centroids

    def __getitem__(self, idx):
        label, feats = self.y[idx].to("cuda"), self.x[idx].to("cuda")
        return label, feats

    def __len__(self):
        return len(self.x)