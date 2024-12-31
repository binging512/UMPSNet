import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .utils import *

class Survival_Dataset(Dataset):
    def __init__(self, cfg, item_list, collect_mode, mode='train') -> None:
        super(Survival_Dataset, self).__init__()
        self.cfg = cfg
        self.collect_mode = collect_mode
        self.mode = mode
        self.item_list = item_list  # training or testing list
        self.dataset_list = ['BLCA', 'BRCA', 'GBMLGG', 'LUAD', 'UCEC']
        
        # Genomic and label file
        slide_data = pd.read_csv(cfg.data_genomic, low_memory=False)
        # Drop the missing slides
        missing_slides_ls = ['TCGA-A7-A6VX-01Z-00-DX2.9EE94B59-6A2C-4507-AA4F-DC6402F2B74F.svs',
                             'TCGA-A7-A0CD-01Z-00-DX2.609CED8D-5947-4753-A75B-73A8343B47EC.svs',
                             'TCGA-HT-7483-01Z-00-DX1.7241DF0C-1881-4366-8DD9-11BF8BDD6FBF.svs',
                             'TCGA-06-0882-01Z-00-DX2.7ad706e3-002e-4e29-88a9-18953ba422bf.svs']
        slide_data.drop(slide_data[slide_data['slide_id'].isin(missing_slides_ls)].index, inplace=True)
        slide_data = slide_data[slide_data['case_id'].isin(item_list)]
        self.label_col = 'survival_months'
        
        # Divide the patents into bins according to the survival length
        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]
        disc_labels, q_bins = pd.qcut(uncensored_df[self.label_col], q=cfg.model_num_classes, retbins=True, labels=False)
        q_bins[-1] = slide_data[self.label_col].max() + 1e-5
        q_bins[0] = slide_data[self.label_col].min() - 1e-5
        disc_labels, q_bins = pd.cut(patients_df[self.label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))
        
        # Make the relationship between the slide_id and case_id
        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})
        self.patient_dict = patient_dict
        
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])
        
        # Separate the cases according to the censorship for WeightedRandomSampler
        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        self.label_dict = label_dict
        # disc_label is the final label for classification
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        self.slide_data = slide_data
        self.cls_ids_prep()
        
        # Read Signatures
        self.signatures = pd.read_csv(cfg.data_signature)
        self.genomic_features = self.slide_data
        
        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))
        
        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq']])
            omic = sorted(series_intersection(omic, self.genomic_features.columns))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]
        
        # Read Status
        self.status_dict = json.load(open(cfg.data_status,'r'))
        
        # Make  data for WeightedRandomSampler
        self.cls_ids_prep()

    def cls_ids_prep(self):
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self,index):
        case_id = self.slide_data['case_id'][index]
        label = self.slide_data['disc_label'][index]
        slide_ids = self.patient_dict[case_id]
        event_time = self.slide_data[self.label_col][index]
        c = self.slide_data['censorship'][index]
        data_belong = self.slide_data['dataset'][index]
        slide_ids = self.patient_dict[case_id]
        dataset_label = self.dataset_list.index(data_belong)
        
        if self.collect_mode in ['transformer_embed']:
            path_dict = read_path_feat(self.cfg.data_patch_feats, slide_ids)
            # Read genomic features and masks
            omics = [self.genomic_features[omic].iloc[index].values for omic in self.omic_names]
            omics_masks = [torch.tensor(~np.isnan(omic)) for omic in omics]  # 1 for no-missing, 0 for missing
            omics_tensors = [torch.tensor(np.nan_to_num(omic)) for omic in omics]  # change NaN to 0
            
            # Set the missing threholds for individual group and overall
            individual_missing_threshold = 0.9
            overall_missing_threshold = 0.9

            # Calculate the missing ratio
            missing_ratios = [1 - omic_mask.float().mean().item() for omic_mask in omics_masks]
            
            # If any individual group missing ratio > individual_missing_threshold
            condition1 = any(ratio > individual_missing_threshold for ratio in missing_ratios)

            # If overall missing ratio > overall_missing_threshold
            condition2 = np.mean(missing_ratios) > overall_missing_threshold

            # Mark the data if it meet the conditions
            omic_missing = torch.tensor(condition1 or condition2, dtype=torch.bool)
            # Read the text data
            text_info = read_text(self.status_dict, case_id)
            
            # Output item dict
            item = {
                'case_id':case_id,
                'path_feats': path_dict['path_feats'], 
                'path_feat_num': path_dict['path_feat_num'],
                'path_pos_ids': path_dict['path_pos_ids'],
                'omics': omics_tensors,
                'mask': omics_masks,
                'omic_missing': omic_missing,  # 省略标记
                'text_info': text_info,
                'label': torch.tensor(int(label)),
                'event_time': torch.tensor(event_time),
                'censorship': torch.tensor(c),
                'data_belong': data_belong,
                'dataset_label': torch.tensor(dataset_label),
                'stage_label': torch.tensor(text_info['stage']),
                'T_stage_label': torch.tensor(text_info['T_stage']),
                'N_stage_label': torch.tensor(text_info['N_stage']),
                'M_stage_label': torch.tensor(text_info['M_stage']),
            }
        else:
            raise NotImplementedError('Dataset collect_mode {} is not implemented!'.format(self.collect_mode))
        
        return item
        
    def __len__(self):
        return len(self.item_list)
