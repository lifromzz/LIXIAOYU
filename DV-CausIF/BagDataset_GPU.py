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
            bag = torch.load(data + "/" + slide_name + ".pt", map_location="cpu")
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

    def build_cluster(self,model,n_clusters,per):

        model.eval()

        topk_0,topk_1=[],[]
        for i,(bag,bag_y) in enumerate(zip(self.x,self.y)):
            bag_num=int(len(bag)*per)
            with torch.no_grad():
                bag = bag.to("cuda")
                output,_,attention = model([bag])
                attention=attention.detach()
                if len(attention.size())>1:
                    attention=attention.squeeze()
                sort_attn,index=attention.sort()
                sample_bag=bag[index[:bag_num]]
                if bag_y==0:
                    topk_0.append(sample_bag)
                else:
                    topk_1.append(sample_bag)

            # index[:bag_num]=0
            # index[bag_num:]=1
            # plot_tsnet(bag,index,f"./tsne/{i}.png")

        centroids_0 = torch.cat(topk_0, dim=0)
        centroids_1 = torch.cat(topk_1, dim=0)
        total_instance=torch.cat([centroids_0,centroids_1],dim=0)
        kmeans = KMeansGPU(n_clusters=n_clusters)
        global_centroids = kmeans.fit(total_instance)


        # kmeans = KMeansGPU(n_clusters=n_clusters)
        # global_centroids_0=kmeans.fit(centroids_0)
        # kmeans = KMeansGPU(n_clusters=n_clusters)
        # global_centroids_1 = kmeans.fit(centroids_1)

        # global_centroids = torch.cat([global_centroids_0,global_centroids_1],dim=0)



        if self.global_centroids ==0:
            return 0.9*self.global_centroids+0.1*global_centroids
        else:
            self.global_centroids=global_centroids
            return global_centroids

        # return global_centroids
    def __getitem__(self, idx):
        label, feats = self.y[idx].to("cuda"), self.x[idx].to("cuda")
        return label, feats

    def __len__(self):
        return len(self.x)