import os
import pandas as pd
from sksurv.metrics import concordance_index_censored
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from datasets import *
from models import *
from utils.tools import *


def validate_fold(cfg, logger, dataloader, model,):
    all_risk_scores, all_censorships, all_event_times = [],[],[]
    all_data_belongs = []
    logger.info('==============Validating==============')
    model.eval()
    for ii, item in enumerate(dataloader):
        if ii%50 == 0:
            logger.info('Validating {}/{} items...'.format(ii, len(dataloader)))
        label = item['label'].type(torch.LongTensor).cuda()
        c = item['censorship'].type(torch.FloatTensor).cuda()
        event_time = item['event_time'].type(torch.FloatTensor).cuda()
        
        with torch.no_grad():
            output = model(item)
            
        risk = -torch.sum(output['S'], dim=1)
        all_risk_scores.append(risk.item())
        all_censorships.append(c.item())
        all_event_times.append(event_time.item())
        all_data_belongs.append(item['data_belong'][0])
    
    result_dict = {'ALL':
        {'risks':all_risk_scores,
         'censorships':all_censorships,
         'event_times': all_event_times}}
    
    for idx in range(len(all_risk_scores)):
        if all_data_belongs[idx] not in result_dict:
            result_dict[all_data_belongs[idx]] = {'risks':[],
                                                  'censorships':[],
                                                  'event_times': []}
        result_dict[all_data_belongs[idx]]['risks'].append(all_risk_scores[idx])
        result_dict[all_data_belongs[idx]]['censorships'].append(all_censorships[idx])
        result_dict[all_data_belongs[idx]]['event_times'].append(all_event_times[idx])
    
    metric_dict = {}
    avg_metric_list = []
    for k, v in result_dict.items():
        c_index = concordance_index_censored((1-np.array(v['censorships'])).astype(bool), 
                                             np.array(v['event_times']), 
                                             np.array(v['risks']), tied_tol=1e-08)[0]
        metric_dict[k] = c_index
        if k != 'ALL':
            avg_metric_list.append(c_index)
    
    avg = np.mean(avg_metric_list)
    metric_dict['AVG'] = avg
        
    model.train()
    return metric_dict, result_dict

def train_fold(cfg, logger, fold_k):
    seed_everything(42)
    best_metric_dict = {'best_ep':0}
    
    split_path = os.path.join(cfg.split_dir, 'splits_{}.csv'.format(fold_k))
    split_df = pd.read_csv(split_path)
    train_list = split_df['train'].dropna().tolist()
    val_list = split_df['val'].dropna().tolist()
    
    logger.info("============Construct Datasets============")
    if cfg.data_name.lower() in ['survival']:
        train_dataset = Survival_Dataset(cfg, train_list, collect_mode = cfg.data_collect_mode, mode=cfg.data_train_mode)
        val_dataset = Survival_Dataset(cfg, val_list, collect_mode = cfg.data_collect_mode, mode=cfg.data_test_mode)
    else:
        raise NotImplementedError('Dataset {} is not implemented!'.format(cfg.data_name))
    
    weights = make_weights_for_balanced_classes_split(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=1, sampler= WeightedRandomSampler(weights, len(weights)), num_workers=1, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    logger.info("============Construct Models============")
    if cfg.model_name.lower() in ['mcat']:
        logger.info('Using model: MCAT')
        model = MCAT(cfg)
    elif cfg.model_name.lower() in ['motcat']:
        logger.info('Using model: MOTCat')
        model = MOTCAT(cfg)
    elif cfg.model_name.lower() in ['umpsnet']:
        logger.info('Using model: UMPSNet!')
        model = UMPSNet(cfg)
    else:
        raise NotImplementedError('Model {} is not implemented!'.format(cfg.model_name))
    
    params = get_parameter_number(model)
    logger.info(params)
    
    if cfg.model_pretrain:
        logger.info('Loading pretrained weight from {}'.format(cfg.model_pretrain))
        pretrain_dict = torch.load(cfg.model_pretrain)
        state_dict = model.state_dict()
        new_state_dict = {}
        for k,v in state_dict.items():
            if k in pretrain_dict and v.shape == pretrain_dict[k].shape:
                new_state_dict[k] = pretrain_dict[k]
            else:
                # logger.info('Pretrained weight not includes {}'.format(k))
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
    
    model = model.cuda()
    
    if cfg.model_loss.lower() in ['nllloss']:
        loss_dict = {'nllloss': NLLSurvLoss(alpha=0.0)}
    elif cfg.model_loss.lower() in ['bnllloss']:
        loss_dict = {'nllloss':Balanced_NLLSurvLoss(alpha=0.0, beta=1.0, n_classes=cfg.model_num_classes)}
    elif cfg.model_loss.lower() in ['nllloss+celoss']:
        loss_dict = {'nllloss': NLLSurvLoss(alpha=0.0), 'celoss':nn.CrossEntropyLoss(ignore_index=-1)}
    else:
        raise NotImplementedError("Loss function {} is not implemented!".format(cfg.model_loss))
    
    optimizer = AdamW(model.parameters(), lr = cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.scheduler.lower() in ['cosine']:
        scheduler = CosineAnnealingLR(optimizer, cfg.epoch_num)
    else:
        scheduler = None
    
    for ep in range(cfg.epoch_num):
        logger.info("==============Training {}/{} Epoches==============".format(ep,cfg.epoch_num))
        cur_iters = 0
        all_risk_scores, all_censorships, all_event_times = np.zeros((len(train_dataloader))), np.zeros((len(train_dataloader))), np.zeros((len(train_dataloader)))
        model.train()
        optimizer.zero_grad()
        for ii, item in enumerate(train_dataloader):
            cur_iters += 1
            label = item['label'].type(torch.LongTensor).cuda()
            c = item['censorship'].type(torch.FloatTensor).cuda()
            event_time = item['event_time'].type(torch.FloatTensor).cuda()
            dataset_label = item['dataset_label'].type(torch.LongTensor).cuda()
            
            output = model(item)
            
            if cfg.model_name.lower() in ['mcat', 'motcat', 'umpsnet']:
                if cfg.model_loss_mode.lower() in ['multimodal']:
                    loss = loss_dict['nllloss'](output['hazards'], output['S'], label, c)
                elif cfg.model_loss_mode.lower() in ['multimodal+dataloss']:
                    loss = loss_dict['nllloss'](output['hazards'], output['S'], label, c)
                    loss_ce_h = loss_dict['celoss'](output['logits_dataset_h'], dataset_label)
                    loss += loss_ce_h
                else:
                    raise NotImplementedError("Loss function {} is not implemented!".format(cfg.mode_loss))
                
            loss.backward()
            
            risk = -torch.sum(output['S'], dim=1)
            all_risk_scores[ii] = risk.item()
            all_censorships[ii] = c.item()
            all_event_times[ii] = event_time.item()
            
            if ii%50 == 0:
                logger.info('Batch: {}/{}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, lr: {}'.format(
                            ii, len(train_dataloader), loss.item(), label.item(), event_time.item(), risk.item(), optimizer.param_groups[0]['lr']))
            if cur_iters % cfg.grad_cum == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        c_index_train = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        logger.info("Epoch: {}, train_c_index: {:.4f}".format(ep, c_index_train))
        
        metric_dict, result_dict = validate_fold(cfg, logger, val_dataloader, model)
        
        logger.info('Epoch: {} Validation Results:'.format(ep))
        for k,v in metric_dict.items():
            if k not in best_metric_dict:
                best_metric_dict[k] = 0
            logger.info("{} dataset, C-index:{:.4f}".format(k, v))
            
        if metric_dict['AVG'] > best_metric_dict['AVG']:
            logger.info("==========New Best Results=========")
            for k,v in metric_dict.items():
                best_metric_dict[k] = v
                logger.info("Dataset: {}, Best C-index:{:.4f}".format(k, v))
            best_metric_dict['best_ep'] = ep
            best_model_save_path = os.path.join(cfg.workspace, 'checkpoints', 'best_model_{}.pth'.format(fold_k))
            logger.info('Best model saved to {}'.format(best_model_save_path))
            os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_save_path)
            best_result_save_path = os.path.join(cfg.workspace, 'results', 'best_result_{}.json'.format(fold_k))
            logger.info('Best result saved to {}'.format(best_result_save_path))
            os.makedirs(os.path.dirname(best_result_save_path), exist_ok=True)
            json.dump(result_dict, open(best_result_save_path, 'w'), indent=2)
            

    logger.info("==============Training complete==============")
    logger.info("Best Result For Fold {}:".format(fold_k))
    for k,v in best_metric_dict.items():
        if k == 'best_ep':
            logger.info("Best epoch: {}".format(v))
        else:
            logger.info("Dataset: {}, Best C-index:{:.4f}".format(k, v))
            
    return best_metric_dict

def train(cfg, logger):
    with torch.cuda.amp.autocast(enabled=False): 
        all_metric_dict = {}
        
        for k in range(cfg.split_start, cfg.split_end):
            logger.info("Start training {}/{} fold...".format(k, cfg.split_end-cfg.split_start))
            metric_dict = train_fold(cfg, logger, k)
            for key, value in metric_dict.items():
                if key in ['best_ep']:
                    pass
                elif key not in all_metric_dict:
                    all_metric_dict[key] = []
                    all_metric_dict[key].append(value)
                else:
                    all_metric_dict[key].append(value)
        
        logger.info('============== Summary ==============')
        datasets = list(all_metric_dict.keys())
        for i in range(cfg.split_start, cfg.split_end):
            metric = ' '.join(["{}: {:.4f}".format(dataset, all_metric_dict[dataset][i-cfg.split_start]) for dataset in datasets])
            logger.info('Fold: {}, {}'.format(i, metric))
        all_metric = ' '.join(["{}: mean: {:.3f}, std: {:.3f}".format(dataset, 
                                                            np.mean(all_metric_dict[dataset]), 
                                                            np.std(all_metric_dict[dataset])) for dataset in datasets])
        ed_metrics = []
        for dataset in datasets:
            if dataset in ['ALL', 'AVG']:
                pass
            else:
                ed_metrics.append(np.mean(all_metric_dict[dataset]))
        ed_metric = np.mean(ed_metrics)
        logger.info(all_metric)
        logger.info("Total: {:.4f}".format(ed_metric))
        
    return 
