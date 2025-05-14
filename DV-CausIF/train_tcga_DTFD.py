import sys
sys.path.append("../")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys, argparse, os, copy, itertools, glob, datetime
import numpy as np
import pandas as pd
import time 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score, recall_score, roc_auc_score, roc_curve,f1_score
import random 
import torch.backends.cudnn as cudnn
import json
torch.multiprocessing.set_sharing_strategy('file_system')
import os

from Models.DTFD.network import get_cam_1d
torch.autograd.set_detect_anomaly(True)

from BagDataset_GPU_DTFD import BagDataset_gpu
from config import get_config


def trainDTFD(args, train_df, classifier, dimReduction, attention, UClassifier,  optimizer0, optimizer1,cluster,causal, \
        criterion=None, numGroup=4, total_instance=4, log_path=''):

    distill = args.distill
    # SlideNames_list, mFeat_list, Label_dict = mDATA_list
    total_loss = 0
    atten_max = 0
    atten_min = 0
    atten_mean = 0
    classifier.train()
    if not args.weight_path:
        dimReduction.train()
    else:
        dimReduction.eval()
    attention.train()
    UClassifier.train()

    instance_per_group = total_instance // numGroup


    numSlides = len(train_df)

    tIDX = list(range(numSlides))
    random.shuffle(tIDX)

    for i, (bag_label, bag_feats) in enumerate(train_df):
        # if i < 265: continue
        bag_label = bag_label.cuda()
        bag_feats = bag_feats.cuda()
        bag_feats = bag_feats.view(-1, args.feats_size)  # n x feat_dim
        tslideLabel = bag_label

        slide_pseudo_feat = []
        slide_sub_preds = []
        slide_sub_labels = []

        tfeat_tensor = bag_feats

        feat_index = list(range(tfeat_tensor.shape[0]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), numGroup)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        for tindex in index_chunk_list:
            slide_sub_labels.append(tslideLabel)
            subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=torch.LongTensor(tindex).cuda())
            tmidFeat = dimReduction(subFeat_tensor)
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            if causal:
                tPredict, bg_feat0, Att_s0 = classifier([tattFeat_tensor,cluster])  ### 1 x 2
            else:
                tPredict, bg_feat0, Att_s0 = classifier([tattFeat_tensor])  ### 1 x 2

            slide_sub_preds.append(tPredict)

            patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.sigmoid(patch_pred_logits)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   ##########################
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            # af_inst_feat = tattFeat_tensor
            af_inst_feat = bg_feat0

            if distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

        ## optimization for the first tier
        
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
        loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
        grad_clipping = 5.0
        if optimizer0:
            optimizer0.zero_grad()
            loss0.backward(retain_graph=True)
            if not args.weight_path:
                torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), grad_clipping)
            torch.nn.utils.clip_grad_norm_(attention.parameters(), grad_clipping)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clipping)
            optimizer0.step()

        ## optimization for the second tier
        # print(slide_pseudo_feat.detach().size())
        
        gSlidePred, bg_feat, Att_s1 = UClassifier(slide_pseudo_feat.detach())
        # print(gSlidePred.size())
        # print(tslideLabel.size())

        # gSlidePred = UClassifier(slide_pseudo_feat)
        loss1 = criterion(gSlidePred.unsqueeze(0), tslideLabel).mean()
        optimizer1.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), grad_clipping)
        optimizer1.step()
        total_loss = total_loss + loss0.item() + loss1.item()

        UAtt = Att_s1
        atten_max = atten_max + UAtt.max().item()
        atten_min = atten_min + UAtt.min().item()
        atten_mean = atten_mean +  UAtt.mean().item()

        sys.stdout.write('\r Training bag [{:}/{:}] bag loss: {:.4f}, attention max:{:.5f}, min:{:.5f}, mean:{:.5f}'.\
            format(i, len(train_df), loss0.item() + loss1.item(), UAtt.max().item(), UAtt.min().item(), UAtt.mean().item()))

    atten_max = atten_max / len(train_df)
    atten_min = atten_min / len(train_df)
    atten_mean = atten_mean / len(train_df)

    with open(log_path,'a+') as log_txt:
            log_txt.write('\n atten_max'+str(atten_max))
            log_txt.write('\n atten_min'+str(atten_min))
            log_txt.write('\n atten_mean'+str(atten_mean))

    return total_loss / len(train_df)


def testDTFD(args, test_df, classifier, dimReduction, attention, UClassifier, \
    criterion,cluster, log_path, epoch, numGroup=4, total_instance=4):

    distill = args.distill
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    instance_per_group = total_instance // numGroup
    gPred_0 = torch.FloatTensor().cuda()
    gt_0 = torch.LongTensor().cuda()
    gPred_1 = torch.FloatTensor().cuda()
    gt_1 = torch.LongTensor().cuda()
    total_loss = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for i,(bag_label,bag_feats) in enumerate(test_df):
            label = bag_label.cpu().numpy()
            # bag_label = bag_label.cuda()
            # bag_feats = bag_feats.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)

            tslideLabel = bag_label
            tfeat = bag_feats
            midFeat = dimReduction(tfeat)
            AA = attention(midFeat, isNorm=False).squeeze(0)  ## N
            allSlide_pred_softmax = []
            num_MeanInference = 1
            for jj in range(num_MeanInference):

                feat_index = list(range(tfeat.shape[0]))
                random.shuffle(feat_index)
                index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                slide_d_feat = []
                slide_sub_preds = []
                slide_sub_labels = []

                for tindex in index_chunk_list:
                    slide_sub_labels.append(tslideLabel)
                    idx_tensor = torch.LongTensor(tindex).cuda()
                    tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                    tAA = AA.index_select(dim=0, index=idx_tensor)
                    tAA = torch.softmax(tAA, dim=0) # n
                    tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                    tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                    tPredict, bg_feat0, Att_s0 = classifier([tattFeat_tensor,cluster])  ### 1 x 2
                    slide_sub_preds.append(tPredict)

                    patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                    patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                    # patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls
                    patch_pred_softmax = torch.sigmoid(patch_pred_logits)  ## n x cls

                    _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                    if distill == 'MaxMinS':
                        topk_idx_max = sort_idx[:instance_per_group].long()
                        topk_idx_min = sort_idx[-instance_per_group:].long()
                        topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                        d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                        slide_d_feat.append(d_inst_feat)
                    elif distill == 'MaxS':
                        topk_idx_max = sort_idx[:instance_per_group].long()
                        topk_idx = topk_idx_max
                        d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                        slide_d_feat.append(d_inst_feat)
                    elif distill == 'AFS':
                        # slide_d_feat.append(tattFeat_tensor)
                        slide_d_feat.append(bg_feat0)

                slide_d_feat = torch.cat(slide_d_feat, dim=0)
                slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
                # test_loss0.update(loss0.item(), numGroup)

                gSlidePred, bag_feat, Att_s1 = UClassifier(slide_d_feat)
                # allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))
                allSlide_pred_softmax.append(torch.sigmoid(gSlidePred)) # [1,1]

            allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
            allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
            gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
            gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

            # loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
            loss1 = criterion(allSlide_pred_softmax.unsqueeze(0), tslideLabel)
            # test_loss1.update(loss1.item(), 1)

            total_loss = total_loss + loss0.item() + loss1.item()

            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss0.item()))
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss1.item()))
            test_labels.extend(label)
            test_predictions.extend([allSlide_pred_softmax.squeeze().cpu().numpy()])

          
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)


    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    test_predictions_= test_predictions>0.5
    acc = accuracy_score(test_labels, test_predictions_)
    cls_report = classification_report(test_labels, test_predictions_, digits=4)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n *****************Threshold by 0.5*****************')
    if args.num_classes==1:
        print('\n', confusion_matrix(test_labels,test_predictions_))
        info = confusion_matrix(test_labels,test_predictions_) 
        with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))
    else:
        for i in range(args.num_classes):
            print('\n', confusion_matrix(test_labels[:,i],test_predictions_[:,i]))
            info = confusion_matrix(test_labels[:,i],test_predictions_[:,i]) 
            with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))
    print('Accuracy', acc)
    print('\n', cls_report)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n Accuracy:'+str(acc))
        log_txt.write('\n'+cls_report)

    # chosing threshold
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n *****************Threshold by optimal*****************')
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
        print(confusion_matrix(test_labels,test_predictions))
        info = confusion_matrix(test_labels,test_predictions)
        with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))
        
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
            print(confusion_matrix(test_labels[:,i],test_predictions[:,i]))
            info = confusion_matrix(test_labels[:,i],test_predictions[:,i])
            with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))

    bag_score = 0
    # average acc of all labels
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)  #ACC
    cls_report = classification_report(test_labels, test_predictions, digits=4)

    # print(confusion_matrix(test_labels,test_predictions))
    print('\n dsmil-metrics: multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
    print('\n', cls_report)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n dsmil-metrics: multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
        log_txt.write('\n' + cls_report)
    if epoch == args.num_epochs-1:
        log_rep = classification_report(test_labels, test_predictions, digits=4,output_dict=True)
        with open(log_path,'a+') as log_txt:
            log_txt.write('{:.2f},{:.2f},{:.2f},{:.2f} \n'.format(log_rep['macro avg']['precision']*100,log_rep['macro avg']['recall']*100,avg_score*100,sum(auc_value)/len(auc_value)*100))

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

def main(dataset,type,fold,feat_size,n_clusters,per,v_dim,cat):
    parser = argparse.ArgumentParser(description='Train DV-CausIF for DTFD')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='DTFD', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--test', action='store_true', help='Test only')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    # parser.add_argument('--dir', type=str,help='directory to save logs')
    parser.add_argument('--dir', type=str,help='directory to save logs')
    parser.add_argument('--static', type=int, nargs='+', default=(0,), help='max:0, mean:1,var:2,min:3')
    parser.add_argument('--weight_path', type=str, default=None, help='directory for loading pretrained model')
    parser.add_argument('--distill', type=str, default='AFS', help='')

    parser.add_argument('--dataset', default=dataset, type=str, help='Dataset folder name')
    parser.add_argument("--type", default=type, choices=['ctrans', 'imagenet'])
    parser.add_argument('--feats_size', default=feat_size, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--fold', default=fold, type=str)


    args = parser.parse_args()
    args.classification_report = os.path.join('train', f"{str(args.model)}_{v_dim}_{str(per)}",
                                              f"causal4_{n_clusters}head4",
                                              f"{args.dataset}_{args.type}_cls_report.csv")

    args.mode = 'a' if os.path.exists(args.classification_report) else 'w'

    # logger
    arg_dict = vars(args)
    dict_json = json.dumps(arg_dict)

    save_path = os.path.join('train', f"{str(args.model)}_{v_dim}_{str(per)}", f"causal4_{n_clusters}head4",
                             str(args.dataset) + '_' + args.type + '_' + args.fold)

    run = len(glob.glob(os.path.join(save_path, '*')))
    save_path = os.path.join(save_path, str(run))
    os.makedirs(save_path, exist_ok=True)
    save_file = save_path + '/config.json'
    with open(save_file,'w+') as f:
        f.write(dict_json)
    log_path = save_path + '/log.txt'
    

    '''
    model 
    1. set require_grad    
    2. choose model and set the trainable params 
    3. load init
    '''
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    from Models.DTFD.network import DimReduction
    from Models.DTFD.Attention import Attention_Gated as Attention
    from Models.DTFD.Attention import Attention_with_Classifier, Classifier_1fc

   # mDim = args.feats_size//2
    mDim=512

    DTFDclassifier_cau = Classifier_1fc(feat_size, args.num_classes,causal=True,v_dim=v_dim,cat=cat).cuda()
    DTFDattentiona_cau = Attention(mDim).cuda()
    DTFDdimReduction_cau = DimReduction(args.feats_size, mDim, numLayer_Res=0).cuda()
    DTFDattCls_cau = Attention_with_Classifier(in_size=mDim, L=mDim, num_cls=args.num_classes).cuda()

    DTFDclassifier = Classifier_1fc(mDim, args.num_classes,causal=False,v_dim=v_dim).cuda()
    DTFDattention = Attention(mDim).cuda()
    DTFDdimReduction = DimReduction(args.feats_size, mDim, numLayer_Res=0).cuda()
    DTFDattCls = Attention_with_Classifier(in_size=mDim, L=mDim, num_cls=args.num_classes).cuda()

    if args.weight_path:
        state_dict_weights = torch.load(args.weight_path)
        msg = DTFDdimReduction.load_state_dict(state_dict_weights['dim_reduction'], strict=False)
        DTFDdimReduction.eval()


    if args.dataset.startswith("tcga"):
        train_data_root, excel_path = get_config("tcga", args.type, None)
        test_data_root = train_data_root

        train_sheet = f"train_fold{args.fold}"
        test_sheet = f"test_fold{args.fold}"

        loss_weight = torch.tensor([[ 1]]).cuda()

    else:
        train_data_root, excel_path = get_config("c16", args.type, "train")
        test_data_root, _ = get_config("c16", args.type, "test")

        train_sheet = "train"
        test_sheet = "test"

        loss_weight = torch.tensor([[ 1]]).cuda()

    trainset = BagDataset_gpu(train_data_root, excel_path, train_sheet, args)
    train_loader = DataLoader(trainset, 1, shuffle=True, num_workers=0)
    testset = BagDataset_gpu(test_data_root, excel_path, test_sheet, args)
    test_loader = DataLoader(testset, 1, shuffle=False, num_workers=0)


    # loss, optim, schduler for DTFD
    criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight).cuda()

    trainable_parameters = []
    trainable_parameters += list(DTFDclassifier.parameters())
    trainable_parameters += list(DTFDattention.parameters())
    if args.weight_path:
        optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=1e-4,  weight_decay=args.weight_decay)
        scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, [int(args.num_epochs/2)], gamma=0.2)
    else:
        trainable_parameters += list(DTFDdimReduction.parameters())
        optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=1e-4,  weight_decay=args.weight_decay)
        scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, [int(args.num_epochs/2)], gamma=0.2)
    optimizer_adam1 = torch.optim.Adam(DTFDattCls.parameters(), lr=1e-4,  weight_decay=args.weight_decay)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, [int(args.num_epochs/2)], gamma=0.2)

    cau_trainable_parameters = []
    cau_trainable_parameters += list(DTFDclassifier_cau.parameters())
    cau_trainable_parameters += list(DTFDattentiona_cau.parameters())
    cau_trainable_parameters += list(DTFDdimReduction_cau.parameters())

    cau_optimizer_adam0 = torch.optim.Adam(cau_trainable_parameters, lr=1e-4, weight_decay=args.weight_decay)
    cau_scheduler0 = torch.optim.lr_scheduler.MultiStepLR(cau_optimizer_adam0, [int(args.num_epochs / 2)], gamma=0.2)

    cau_optimizer_adam1 = torch.optim.Adam(DTFDattCls_cau.parameters(), lr=1e-4, weight_decay=args.weight_decay)
    cau_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(cau_optimizer_adam1, [int(args.num_epochs / 2)], gamma=0.2)

    best_score = 0


    for epoch in range(1, args.num_epochs):
        start_time = time.time()
        train_loss_bag = trainDTFD(args, train_loader,
                                   DTFDclassifier, DTFDdimReduction, DTFDattention, DTFDattCls,
                                   optimizer_adam0, optimizer_adam1, causal=False,cluster=None,
                                   criterion=criterion,log_path=log_path)
        cluster=trainset.build_cluster(DTFDdimReduction,DTFDattention,n_clusters,per)

        train_loss_bag = trainDTFD(args, train_loader,
                                   DTFDclassifier_cau, DTFDdimReduction_cau, DTFDattentiona_cau, DTFDattCls_cau,
                                   cau_optimizer_adam0, cau_optimizer_adam1, causal=True, cluster=cluster,
                                   criterion=criterion, log_path=log_path)

        print('epoch time:{}'.format(time.time()- start_time))


        test_loss_bag, avg_score, aucs, thresholds_optimal = \
            testDTFD(args, test_loader,
                     DTFDclassifier_cau, DTFDdimReduction_cau, DTFDattentiona_cau,DTFDattCls_cau,
                     criterion,cluster, log_path,epoch)
        
        info = 'Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: '%(epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))+'\n'
        with open(log_path,'a+') as log_txt:
            log_txt.write(info)
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 

        scheduler0.step()
        scheduler1.step()
        cau_scheduler0.step()
        cau_scheduler1.step()

        current_score = (sum(aucs) + avg_score)/2
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(save_path, str(run+1)+'org.pth')
            tsave_dict = {
                'classifier': DTFDclassifier.state_dict(),
                'dim_reduction': DTFDdimReduction.state_dict(),
                'attention': DTFDattention.state_dict(),
                'att_classifier': DTFDattCls.state_dict()
            }
            torch.save(tsave_dict, save_name)

            save_name = os.path.join(save_path, str(run + 1) + 'cau.pth')
            tsave_dict = {
                'classifier': DTFDclassifier_cau.state_dict(),
                'dim_reduction': DTFDdimReduction_cau.state_dict(),
                'attention': DTFDattentiona_cau.state_dict(),
                'att_classifier': DTFDattCls_cau.state_dict()
            }
            torch.save(tsave_dict, save_name)

            with open(log_path,'a+') as log_txt:
                info = 'Best model saved at: ' + save_name +'\n'
                log_txt.write(info)
                info = 'Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal))+'\n'
                log_txt.write(info)
            print('Best model saved at: ' + save_name)
            print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
        if epoch == args.num_epochs-1:
            save_name = os.path.join(save_path, 'last_org.pth')
            tsave_dict = {
                'classifier': DTFDclassifier.state_dict(),
                'dim_reduction': DTFDdimReduction.state_dict(),
                'attention': DTFDattention.state_dict(),
                'att_classifier': DTFDattCls.state_dict()
            }
            torch.save(tsave_dict, save_name)

            tsave_dict = {
                'classifier': DTFDclassifier_cau.state_dict(),
                'dim_reduction': DTFDdimReduction_cau.state_dict(),
                'attention': DTFDattentiona_cau.state_dict(),
                'att_classifier': DTFDattCls_cau.state_dict()
            }
            causal_save_name=os.path.join(save_path, 'last_cau.pth')
            torch.save(tsave_dict, causal_save_name)
    log_txt.close()

if __name__ == '__main__':
    for i in range(1):
        for n_clusters in [16]:
            for i in range(5):
                main("tcga", "imagenet", str(i), 512, n_clusters, 0.5, 512, cat=False)
            for i in range(5):
                main("tcga", "ctrans", str(i), 768, n_clusters, 0.1, 512, cat=False)
            for i in range(5):
                main( "c16", "imagenet", str(i), 512, n_clusters, 0.1, 512, cat=False)
            for i in range(5):
                main( "c16", "ctrans", str(i), 768, n_clusters, 0.1, 512, cat=False)

