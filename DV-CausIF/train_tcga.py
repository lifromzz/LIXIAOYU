import sys

import pandas as pd

sys.path.append("../")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import sys, argparse, os, copy, itertools, glob, datetime
import numpy as np

import time 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score, recall_score, roc_auc_score, roc_curve,f1_score
import random 
import torch.backends.cudnn as cudnn
import json
# torch.multiprocessing.set_sharing_strategy('file_system')
import os

from BagDataset_GPU import BagDataset_gpu
from config import get_config
import gc



def train(train_df, milnet, criterion, optimizer,clusters, args, log_path,causal, epoch=0):
    milnet.train()
    total_loss = 0
    atten_max = 0
    atten_min = 0
    atten_mean = 0
    
    for i,(bag_label,bag_feats) in enumerate(train_df):
        bag_label = bag_label.cuda()
        bag_feats = bag_feats.cuda()
        bag_feats = bag_feats.view(-1, args.feats_size)  # n x feat_dim
        optimizer.zero_grad()
        if args.model == 'dsmil':
            if causal:
                ins_prediction, bag_prediction, attention= milnet([bag_feats,clusters])
            else:
                ins_prediction, bag_prediction, attention= milnet([bag_feats])

            max_prediction, _ = torch.max(ins_prediction, 0)

            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss

        elif (args.model =='abmil') or (args.model == "ilra"):
            if causal:
                bag_prediction, _, attention = milnet([bag_feats,clusters])
            else:
                bag_prediction, _, attention = milnet([bag_feats])
            loss =  criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))

        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        atten_max = atten_max + attention.max().item()
        atten_min = atten_min + attention.min().item()
        atten_mean = atten_mean +  attention.mean().item()
        
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f, attention max:%.4f, min:%.4f, mean:%.4f' % (i, len(train_df), loss.item(), 
                        attention.max().item(), attention.min().item(), attention.mean().item()))
    atten_max = atten_max / len(train_df)
    atten_min = atten_min / len(train_df)
    atten_mean = atten_mean / len(train_df)
    with open(log_path,'a+') as log_txt:
            log_txt.write('\n atten_max'+str(atten_max))
            log_txt.write('\n atten_min'+str(atten_min))
            log_txt.write('\n atten_mean'+str(atten_mean))
    return total_loss / len(train_df)


def test(test_df, milnet, criterion, optimizer,clusters, args, log_path, epoch):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for i,(bag_label,bag_feats) in enumerate(test_df):
            label = bag_label.cpu().numpy()

            bag_feats = bag_feats.view(-1, args.feats_size)
            if args.model == 'dsmil':
                ins_prediction, bag_prediction, _ = milnet([bag_feats,clusters])
                max_prediction, _ = torch.max(ins_prediction, 0)  

            elif args.model in ['abmil','ilra']:
                bag_prediction, _, _ =  milnet([bag_feats,clusters])
                max_prediction = bag_prediction

            test_labels.extend(label)
            if args.average:   # notice args.average here
                test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
                
            else: test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])


    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)

    with open(log_path, 'a+') as log_txt:
        log_txt.write('\n *****************Threshold by optimal*****************')
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
        print('\n')
        print(confusion_matrix(test_labels, test_predictions))
        info = confusion_matrix(test_labels, test_predictions)
        with open(log_path, 'a+') as log_txt:
            log_txt.write('\n' + str(info))
    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
            print(confusion_matrix(test_labels[:, i], test_predictions[:, i]))
            info = confusion_matrix(test_labels[:, i], test_predictions[:, i])
            with open(log_path, 'a+') as log_txt:
                log_txt.write('\n' + str(info))
    bag_score = 0
    # average acc of all labels
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_df)  # ACC
    cls_report = classification_report(test_labels, test_predictions, digits=4)
    print('\n  multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score * 100, sum(auc_value) / len(auc_value) * 100))
    print('\n', cls_report)
    with open(log_path, 'a+') as log_txt:
        log_txt.write(
            '\n multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score * 100, sum(auc_value) / len(auc_value) * 100))
        log_txt.write('\n' + cls_report)
    if epoch == args.num_epochs - 1:
        precision = precision_score(test_labels, test_predictions, average='macro')
        recall = recall_score(test_labels, test_predictions, average='macro')
        f1 = f1_score(test_labels, test_predictions, average='macro')

        result_dict = {'Acc': [avg_score * 100], 'auc': [sum(auc_value) / len(auc_value) * 100],
                       'precision': [precision * 100],
                       'recall': [recall * 100],
                       'f1': [f1 * 100]}
        df = pd.DataFrame(result_dict)

        df.to_csv(args.classification_report, mode=args.mode, index=True, header=True if args.mode == "w" else False)
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal


def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        if sum(label)==0:
            continue
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main(model,dataset,type,fold,feats_size,n_clusters,per,v_dim,cat):
    parser = argparse.ArgumentParser(description='Train DV-CausIF for abmil , dsmil and ILRA-MIL')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--weight_decay_conf', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--test', action='store_true', help='Test only')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')


    parser.add_argument('--dataset', default=dataset, type=str, help='Dataset folder name')
    parser.add_argument("--type",default=type,choices=['ctrans','imagenet'])
    parser.add_argument('--feats_size', default=feats_size, type=int, help='Dimension of the feature size [512]')

    parser.add_argument('--fold', default=fold, type=str)
    parser.add_argument('--model', default=model, type=str, help='MIL model [admil, dsmil]')
    parser.add_argument('--teacher', default=f'../baseline/{type}/{dataset}_{model}_no_fulltune_{fold}/0/1.pth', type=str)


    args = parser.parse_args()

    args.classification_report = os.path.join('train', f"{str(args.model)}_{v_dim}_{str(per)}_causal4_ffn_0.3_head4_mom0.9_q+w(m)",
                                              f"clusters_irrelevant_{n_clusters}",
                                              f"{args.dataset}_{args.type}_cls_report.csv")

    args.mode= 'a' if os.path.exists(args.classification_report) else 'w'

    arg_dict = vars(args)
    dict_json = json.dumps(arg_dict)

    save_path = os.path.join('train', f"{str(args.model)}_{v_dim}_{str(per)}_causal4_ffn_0.3_head4_mom0.9_q+w(m)",
                             f"clusters_irrelevant_{n_clusters}",
                             str(args.dataset) + '_' + args.type + '_' + args.fold)

    run = len(glob.glob(os.path.join(save_path, '*')))
    save_path = os.path.join(save_path, str(run))
    os.makedirs(save_path, exist_ok=True)
    save_file = save_path + '/config.json'
    with open(save_file,'w+') as f:
        f.write(dict_json)
    log_path = save_path + '/log.txt'
    

    # seed 
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    '''
    model 
    1. set require_grad    
    2. choose model and set the trainable params 
    3. load init
    '''
    if args.model == 'dsmil':
        import our_topk.Models.dsmil as mil
        i_classifier_cau = mil.FCLayer(in_size=v_dim).cuda()
        b_classifier_cau = mil.BClassifier(input_size=v_dim,cluster_dim=args.feats_size, causal=True,v_dim=v_dim,cat=cat).cuda()
        milnet_causal = mil.MILNet(args.feats_size,v_dim,i_classifier_cau, b_classifier_cau,causal=True).cuda()

        i_classifier = mil.FCLayer(in_size=v_dim).cuda()
        b_classifier= mil.BClassifier(input_size=v_dim,
                                           cluster_dim=args.feats_size,

                                           causal=False,v_dim=v_dim).cuda()
        milnet=mil.MILNet(args.feats_size,v_dim,i_classifier, b_classifier,causal=False).cuda()
    elif args.model == 'abmil':
        import Models.abmil as mil
        milnet_causal = mil.Attention(in_size=args.feats_size,causal=True,v_dim=v_dim,cat=cat).cuda()
        milnet = mil.Attention(in_size=args.feats_size,causal=False,v_dim=v_dim).cuda()

    elif args.model== "ilra":
        from Models.ILRA import ILRA
        milnet = ILRA(feat_dim=args.feats_size,hidden_feat=v_dim).cuda()
        milnet_causal = ILRA(feat_dim=args.feats_size,causal=True,cat=cat).cuda()


    if args.dataset.startswith("tcga"):
        train_data_root, excel_path = get_config("tcga", args.type,None)
        test_data_root=train_data_root

        train_sheet=f"train_fold{args.fold}"
        test_sheet=f"test_fold{args.fold}"


    else:
        train_data_root, excel_path = get_config("c16", args.type, "train")
        test_data_root, _ = get_config("c16", args.type, "test")

        train_sheet = "train"
        test_sheet = "test"
        
    trainset =  BagDataset_gpu(train_data_root,excel_path, train_sheet,args)
    train_loader = DataLoader(trainset,1, shuffle=True, num_workers=0)
    testset =  BagDataset_gpu(test_data_root,excel_path, test_sheet,args)
    test_loader = DataLoader(testset,1, shuffle=False, num_workers=0)

    # sanity check begins here
    print('*******sanity check *********')
    for k,v in milnet.named_parameters():
        if v.requires_grad == True:
            print(k)

     # loss, optim, schduler
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, milnet.parameters()), 
                                lr=args.lr, betas=(0.5, 0.9), 
                                weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

    optimizer_causal = torch.optim.Adam(filter(lambda p: p.requires_grad, milnet_causal.parameters()),
                                 lr=args.lr, betas=(0.5, 0.9),
                                 weight_decay=args.weight_decay)

    scheduler_causal = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_causal, args.num_epochs, 0.000005)
    best_score = 0


    for epoch in range(1, args.num_epochs):
        start_time = time.time()
        train_loss_bag = train(train_loader, milnet, criterion, optimizer, None,args, log_path,causal=False, epoch=epoch-1) # iterate all bags

        centers=trainset.build_cluster(milnet,n_clusters,per)
        train_loss_bag = train(train_loader, milnet_causal, criterion, optimizer_causal, centers,args, log_path,causal=True, epoch=epoch-1) # iterate all bags

        print('epoch time:{}'.format(time.time()- start_time))

        test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet_causal, criterion, optimizer_causal,centers, args, log_path, epoch)
        info = 'Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % (
        epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join(
            'class-{}>>{}'.format(*k) for k in enumerate(aucs)) + '\n'
        with open(log_path, 'a+') as log_txt:
            log_txt.write(info)
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' %
              (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join(
            'class-{}>>{}'.format(*k) for k in enumerate(aucs)))
        scheduler.step()
        scheduler_causal.step()

        current_score = (sum(aucs) + avg_score) / 2
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(save_path, str(run+1)+'org.pth')
            torch.save(milnet.state_dict(), save_name)

            causal_save_name=os.path.join(save_path, str(run+1)+'cau.pth')
            torch.save(milnet_causal.state_dict(), causal_save_name)


        if epoch == args.num_epochs-1:
            save_name = os.path.join(save_path, 'last_org.pth')
            torch.save(milnet.state_dict(), save_name)

            causal_save_name=os.path.join(save_path, 'last_cau.pth')

            torch.save(milnet_causal.state_dict(), causal_save_name)
    log_txt.close()


if __name__ == '__main__':


    for model_name in ["abmil"]:  #"abmil","dsmil","ilra"
        for n in [0.3]: #2,4,8,,0.3,0.4,0.5
            for i in range(5):
                main(model_name,"c16", "imagenet", str(i), 512, 16, 0.3, 512,cat=False)
            for i in range(5):
                main(model_name,"c16", "ctrans", str(i), 768, 16, 0.3, 512,cat=False)

            for i in range(5):
                main(model_name,"tcga", "imagenet", str(i), 512, 16, 0.3, 512,cat=False)
            for i in range(5):
                main(model_name,"tcga", "ctrans", str(i), 768, 16, 0.3, 512,cat=False)

