
dataset_root="/root/autodl-tmp/Datasets/"
def get_config(dataname,type,train):
    if "tcga" in dataname:
        if "ctrans" in type:
            data_root = dataset_root+"tcga/tcga_ctrans_baseline_level0_256"
        else:
            data_root = dataset_root+"tcga/tcga_ImageNet_r18_l4_level0_256"

        split_path = dataset_root+"/tcga/file_labels_seed0.xlsx"
    else:
        if "ctrans" in type:
            data_root = dataset_root + "c16/Camelyon16_ctrans_baseline_level0_256/"+train
        else:
            data_root = dataset_root + "c16/Camelyon16_ImageNet_r18_l4_level0_256/"+train
        split_path = dataset_root + "c16/c16.xlsx"
    return data_root,split_path
