import os
import h5py
import numpy as np
import torch
from sklearn.cluster import KMeans

def read_text(status_dict, case_id:str):
    # Demographic
    race = status_dict[case_id]['demographic']['race']
    age = status_dict[case_id]['demographic']['age']
    if age == -1:
        age = 'unknown'
        age_tensor = -1.0
    else:
        age_tensor = age/100
        
    gender = status_dict[case_id]['demographic']['gender']
    if gender.lower() in ['female']:
        sex_prompt = 'she'
        gender_prompt = 'woman'
        sex_tensor = 0
    else:
        sex_prompt = 'he'
        gender_prompt = 'man'
        sex_tensor = 1
    
    if race.lower() in ['white']:
        race_prompt, race_tensor = 'white', 0
    elif race.lower() in ['black or african american']:
        race_prompt, race_tensor = 'black or african american', 1
    elif race.lower() in ['asian']:
        race_prompt, race_tensor = 'asian', 2
    elif race.lower() in ['american indian or alaska native']:
        race_prompt, race_tensor = 'american indian or alaska native', 3
    elif race.lower() in ['native hawaiian or other pacific islander']:
        race_prompt, race_tensor = 'native hawaiian or other pacific islander', 4
    elif race.lower() in ['not reported']:
        race_prompt, race_tensor = 'unknown', -1
        
    demographic_text = "The patient is a {}-year-old {} race {}.".format(age, race_prompt, gender_prompt)
    
    # Diagnosis
    primary_diagnosis = status_dict[case_id]['diagnosis']['primary_diagnosis']
    stage = status_dict[case_id]['diagnosis']['ajcc_pathologic_stage']
    if stage.lower() in ['stage i', 'stage ia', 'stage ib']:
        stage_prompt, stage_tensor='early stage', 0
    elif stage.lower() in ['stage ii', 'stage iia', 'stage iib']:
        stage_prompt, stage_tensor= 'early to mid-term', 1
    elif stage.lower() in ['stage iii', 'stage iiia', 'stage iiib']:
        stage_prompt, stage_tensor='middle and late stage', 2
    elif stage.lower() in ['stage iv', 'stage iva', 'stage ivb']:
        stage_prompt, stage_tensor='late stage', 3
    else:
        stage_prompt, stage_tensor='unknown stage', -1
        
    T_stage = status_dict[case_id]['diagnosis']['ajcc_pathologic_t']
    if T_stage.lower() in ['t0']:
        t_stage_prompt, t_stage_tensor='no tumor is found', 0
    elif T_stage.lower() in ['t1', 't1a', 't1b', 't1c']:
        t_stage_prompt, t_stage_tensor='tumors are samll, no local vascular invasion', 1
    elif T_stage.lower() in ['t2', 't2a', 't2b', 't2c']:
        t_stage_prompt, t_stage_tensor='tumors are large, but no local vascular invasion', 2
    elif T_stage.lower() in ['t3', 't3a', 't3b', 't3c']:
        t_stage_prompt, t_stage_tensor='tumors are large, with local vascular invasion', 3
    elif T_stage.lower() in ['t4', 't4a', 't4b', 't4c']:
        t_stage_prompt, t_stage_tensor='many large tumors are found, with vast vascular invasion', 4
    else:
        t_stage_prompt, t_stage_tensor='tumor status is unknown', -1
        
    N_stage = status_dict[case_id]['diagnosis']['ajcc_pathologic_n']
    if N_stage.lower() in ['n0', 'n0 (i-)', 'n0 (i+)']:
        n_stage_prompt, n_stage_tensor='no regional lymph node metastasis', 0
    elif N_stage.lower() in ['n1', 'n1a', 'n1b']:
        n_stage_prompt, n_stage_tensor='small parts of lymph node metastasis', 1
    elif N_stage.lower() in ['n2', 'n2a', 'n2b']:
        n_stage_prompt, n_stage_tensor='medium parts of lymph node metastasis', 2
    elif N_stage.lower() in ['n3', 'n3a', 'n3b']:
        n_stage_prompt, n_stage_tensor='large parts of lymph node metastasis', 3
    else:
        n_stage_prompt, n_stage_tensor='unknown lymph node metastasis', -1
    
    M_stage = status_dict[case_id]['diagnosis']['ajcc_pathologic_m']
    if M_stage.lower() in ['m0']:
        m_stage_prompt, m_stage_tensor='no tumor transfer', 0
    elif M_stage.lower() in ['m1']:
        m_stage_prompt, m_stage_tensor='with tumor transfer', 1
    else:
        m_stage_prompt, m_stage_tensor='unknown tumor transfer', -1
        
    diagnosis_text = '{} has {} at {}. {}, {}, {}.'.format(
        sex_prompt.capitalize(), primary_diagnosis.lower(), stage_prompt,
        t_stage_prompt.capitalize(), n_stage_prompt,  m_stage_prompt)
    
    # Treatments
    pharmaceutical_therapy = status_dict[case_id]['treatments']['Pharmaceutical Therapy']
    radiation_therapy = status_dict[case_id]['treatments']['Radiation Therapy']
    if pharmaceutical_therapy.lower() in ['yes'] and radiation_therapy.lower() in ['yes']:
        treatment_prompt = 'Both pharmaceutical and radiation therapies are applied.'
        p_tensor, r_tensor = 1,1
    elif pharmaceutical_therapy.lower() in ['yes'] and radiation_therapy.lower() in ['no', 'not reported']:
        treatment_prompt = 'Only pharmaceutical therapy is applied.'
        p_tensor, r_tensor = 1,0
    elif pharmaceutical_therapy.lower() in ['no','not reported'] and radiation_therapy.lower() in ['yes']:
        treatment_prompt = 'Only radiation therapy is applied.'
        p_tensor, r_tensor = 0,1
    else:
        treatment_prompt = 'No pharmaceutical or radiation therapy is applied.'
        p_tensor, r_tensor = 0,0
    
    treatment_text = treatment_prompt
    
    cancer_name = status_dict[case_id]['dataset']
    if cancer_name.lower() in ['blca']:
        cancer_text = "This is a patient who has bladder urothelial carcinoma."
    elif cancer_name.lower() in ['brca']:
        cancer_text = 'This is a patient who has breast invasive carcinoma.'
    elif cancer_name.lower() in ['gbmlgg']:
        cancer_text = "This is a patient who has glioblastoma multiforme or brain lower grade glioma."
    elif cancer_name.lower() in ['luad']:
        cancer_text = "This is a patient who has lung adenocarcinoma."
    elif cancer_name.lower() in ['ucec']:
        cancer_text = "This is a patient who has uterine corpus endometrial carcinoma."
    else:
        raise NotImplementedError("cancer name {} is not implemented!".format(cancer_text))
    
    text_info = {
        'demographic_text': demographic_text,
        'diagnosis_text': diagnosis_text,
        'treatment_text': treatment_text,
        'cancer_text': cancer_text,
        'stage': stage_tensor,
        'T_stage': t_stage_tensor,
        'N_stage': n_stage_tensor,
        'M_stage': m_stage_tensor
    }
    
    return text_info


def read_path_feat(feat_dir, slide_ids):
    feat_list, coord_list = [], []
    pos_dir = feat_dir.replace('pt_files', 'pos_files')
    
    for ii, slide_id in enumerate(slide_ids):
        wsi_path = os.path.join(feat_dir, '{}.pt'.format(slide_id.rstrip('.svs')))
        wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
        feat_list.append(wsi_bag)
        
        pos_path = os.path.join(pos_dir, '{}.pth'.format(slide_id.rstrip('.svs')))
        pos_bag = torch.load(pos_path, map_location=torch.device('cpu'))
        coord_list.append(pos_bag)
        
    path_features = torch.cat(feat_list, dim=0)
    pos_features = []
    for i in range(len(coord_list)):
        pos_prefix = torch.ones((len(coord_list[i]),1))*i
        pos_ids = torch.cat((pos_prefix,coord_list[i]), dim=1)    # Nx3 [(0,1.0,1.0),...,(1,2,3)]
        pos_features.append(pos_ids)
    path_positions = torch.cat(pos_features, dim=0)
    
    path_dict = {'path_feats': path_features,
                 'path_pos_ids': path_positions,
                 'path_feat_num': len(slide_ids)}

    return path_dict

