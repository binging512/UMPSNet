import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

import ot

from .utils import *
from .clip import clip
# torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_name='ViT-L/14', out_channels=64) -> None:
        super(TextEncoder,self).__init__()
        self.clip, self.clip_preprocess = clip.load(clip_name, download_root='./pretrained')
        self.adapter = Adapter(c_in=768)
        self.adapter_alpha = nn.parameter.Parameter(torch.tensor(0.2))
        self.adapter_projection = nn.Sequential(nn.Linear(768, out_channels), nn.ReLU(inplace=True))
        
        for name, param in self.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)
        
    def forward(self, texts):
        text_tokens = clip.tokenize(texts).cuda()
        text_feat = self.clip.encode_text(text_tokens).float()
        text_adapt = self.adapter(text_feat)
        text_feat = self.adapter_alpha*text_adapt + (1-self.adapter_alpha)*text_feat
        text_feat = self.adapter_projection(text_feat)
        
        return text_feat


class OT_Attn_assem(nn.Module):
    def __init__(self, impl='pot-uot-l2', ot_reg=0.1, ot_tau=0.5) -> None:
        super().__init__()
        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau
        print("ot impl: ", impl)

    def normalize_feature(self, x):
        x = x - x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, weight1, weight2, epsilon=1e-8):
        """
        Params:
            weight1 : (N, D)
            weight2 : (M, D)
        
        Return:
            flow : (N, M)
            dist : (1, )
        """

        if self.impl == "pot-sinkhorn-l2":
            self.cost_map = torch.cdist(weight1, weight2) ** 2  # (N, M)
            """cold"""
            self.cost_map = torch.clamp(self.cost_map, max=1e10)  # 防止代价矩阵数值过大
            
            src_weight = weight1.sum(dim=1) / (weight1.sum() + epsilon)
            dst_weight = weight2.sum(dim=1) / (weight2.sum() + epsilon)
            
            cost_map_detach = self.cost_map.detach()
            flow = ot.sinkhorn(a=src_weight.detach(), b=dst_weight.detach(), 
                               M=cost_map_detach / (cost_map_detach.max() + epsilon), reg=self.ot_reg)
            dist = self.cost_map * flow
            dist = torch.sum(dist)
            return flow, dist
        
        elif self.impl == "pot-uot-l2":
            a, b = torch.from_numpy(ot.unif(weight1.size(0)).astype('float64')).to(weight1.device), torch.from_numpy(ot.unif(weight2.size(0)).astype('float64')).to(weight2.device)
            self.cost_map = torch.cdist(weight1, weight2) ** 2  # (N, M)
            self.cost_map = torch.clamp(self.cost_map, max=1e10)  # 防止代价矩阵数值过大
            
            cost_map_detach = self.cost_map.detach()
            M_cost = cost_map_detach / (cost_map_detach.max() + epsilon)
            
            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b, 
                                                           M=M_cost.double(), reg=self.ot_reg, reg_m=self.ot_tau)
            flow = flow.type(torch.FloatTensor).cuda()
            
            dist = self.cost_map * flow  # (N, M)
            dist = torch.sum(dist)  # (1,) float
            if torch.any(torch.isnan(flow)):
                print(f"NaN detected in flow")
                print(f"Flow: {flow}")
            
            if torch.any(torch.isnan(dist)):
                print(f"NaN detected in dist")
                print(f"Dist: {dist}")
            return flow, dist
        
        else:
            raise NotImplementedError

    def forward(self, x, y):
        '''
        x: (N, 1, D)
        y: (M, 1, D)
        '''
        x = x.squeeze()
        y = y.squeeze()
        
        x = self.normalize_feature(x)
        y = self.normalize_feature(y)
        
        pi, dist = self.OT(x, y)
        return pi.T.unsqueeze(0).unsqueeze(0), dist

    
class PositionalEncoding(nn.Module):
    """实现位置编码的模块，帮助 Transformer 感知序列中各个元素的相对位置"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个大小为 (max_len, d_model) 的位置编码张量
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        

class GeneTransformerEmbedder(nn.Module):
    def __init__(self, input_dim=1, d_model=256, nhead=8, num_layers=2, dropout=0.1):
        """
        input_dim: 每个基因点位的特征维度，默认为 1（每个基因点位是一个浮点数）。
        d_model: Transformer 隐藏层的维度。
        nhead: 多头注意力机制中的头数。
        num_layers: Transformer 编码器的层数。
        dropout: dropout 概率。
        """
        super(GeneTransformerEmbedder, self).__init__()
        # 基因点位的线性投影层，将输入投影到 Transformer 的输入维度
        self.input_projection = nn.Linear(input_dim, d_model)
        # 位置编码，帮助 Transformer 感知基因点位的相对位置
        self.position_encoding = PositionalEncoding(d_model, dropout)   
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=d_model * 2, 
                                                   dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 聚合层，用于将变长的基因数据转化为固定维度的表示（例如平均池化）
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        # 假设 self.omic_sizes 是每组基因的基因点位数量
        
    def forward(self, gene_data, mask):
        """
        gene_data: 基因数据，形状为 (num_genes, input_dim)，其中 num_genes 是基因点位的数量，
                   input_dim 通常为 1 表示每个基因点位的一个浮点数。
        mask: mask 矩阵，用来屏蔽缺失的基因点位。形状为 (num_genes,)，值为 0 或 1。
        """
        # 将基因点位特征通过线性投影层投影到 Transformer 的输入维度
        gene_features = self.input_projection(gene_data)  # Shape: (num_genes, d_model)
        
        # 添加位置编码
        gene_features = self.position_encoding(gene_features.unsqueeze(1)).squeeze(1)
        
        # 生成 mask，mask 需要扩展到 (batch_size, num_genes) 的形状
        # mask 的 0 表示无效位置，1 表示有效位置
        src_key_padding_mask = (mask == 0)  # 需要转为 bool 类型
        
        # 将基因点位特征输入 Transformer 编码器，同时传入 mask
        gene_transformed = self.transformer_encoder(gene_features, src_key_padding_mask=src_key_padding_mask)  # Shape: (num_genes, d_model)
        
        # 使用自适应池化将变长基因数据聚合为固定长度的向量
        gene_pooled = self.pooling_layer(gene_transformed.transpose(1, 0).unsqueeze(0)).squeeze()
        
        return gene_pooled  # 输出形状为 (d_model,)


class UniModule_Guided(nn.Module):
    def __init__(self, size, nhead, dropout) -> None:
        super(UniModule_Guided, self).__init__()
        layer = nn.TransformerDecoderLayer(d_model=size, nhead=nhead, dim_feedforward=512, dropout=dropout, activation='relu', batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.mm = nn.Sequential(*[nn.Linear(size*4, size), nn.ReLU(), nn.Linear(size, size), nn.ReLU()])
    
    def forward(self, h_omic, h_path, h_text_bag):
        h_text_bag = h_text_bag.permute(1,0,2)    # 4x1xC ->1x4xC
        h = torch.cat([h_omic, h_path], dim=1)    # 1x2xC
        h = self.decoder(h_text_bag, h)           # 1x4xC
        h = h.view(1, 1, -1)                      # 1x1x(4xC)
        cls_feat = self.mm(h)                     # 1x1xC
        return cls_feat    


class ExpertTransformer_guided(nn.Module):
    def __init__(self, n_classes, size, omic_sizes, ot_reg=0.1, ot_tau=0.5, ot_impl="pot-uot-l2", dropout=0.25, d_model=256, nhead=8, dropout_decoder=0.25):
        super(ExpertTransformer_guided, self).__init__()

        self.mm = UniModule_Guided(size[2], nhead=8, dropout=dropout)
        
        self.n_classes = n_classes
        self.classifier = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Linear(size[2], self.n_classes)])
    
    def forward(self, h_path_res, h_omic_res, h_text_bag):  
        
        h = self.mm(h_path=h_path_res, h_omic=h_omic_res, h_text_bag=h_text_bag)
        
        logits = self.classifier(h)    
        
        return {"logits":logits}


class MixtureOfExperts(nn.Module):
    def __init__(self, n_classes, size, omic_sizes, num_experts=3,ot_reg=0.1, ot_tau=0.5, ot_impl="pot-uot-l2", dropout=0.25, d_model=256, nhead=8, dropout_decoder=0.25):
        super(MixtureOfExperts, self).__init__()
        
        """guided"""
        self.text_encoder = TextEncoder(clip_name='ViT-L/14', out_channels = size[2])
        self.experts = nn.ModuleList([
            ExpertTransformer_guided(size=size, omic_sizes=omic_sizes, n_classes=n_classes, ot_reg=0.1, ot_tau=0.5, ot_impl="pot-uot-l2", dropout=0.25, d_model=256, nhead=8, dropout_decoder=0.1) for _ in range(num_experts)
        ])
        
        self.combine_weight = nn.Parameter(torch.randn(num_experts, d_model*2))
        self.disease_to_id = {
            "BLCA":0,
            "BRCA":1,
            "GBMLGG":2,
            "LUAD":3,
            "UCEC":4
        }
        self.disease_vocab_size = len(self.disease_to_id)
        self.disease_embedding = nn.Embedding(self.disease_vocab_size, d_model)
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)
        
        self.sig_networks = nn.ModuleList([GeneTransformerEmbedder(input_dim=1) for _ in omic_sizes])
        self.embed_size = size[2]
        
        ### OT-based Co-attention
        self.path_coattn = OT_Attn_assem(impl=ot_impl,ot_reg=ot_reg,ot_tau=ot_tau)
        self.omic_coattn = OT_Attn_assem(impl=ot_impl,ot_reg=ot_reg,ot_tau=ot_tau)

        ### Path Transformer + Attention Head
        path_encoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerDecoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Omic Transformer + Attention Head
        omic_encoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerDecoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        self.mm = UniModule_Guided(size[2], nhead=8, dropout=dropout)
        
    def forward(self, item):
        x_path = item['path_feats'][0].cuda()
        x_omic = [omic[0].to(device).float() for omic in item['omics']]
        x_omic_mask = [omic_mask[0].to(device).float() for omic_mask in item['mask']]
        x_demo_text = item['text_info']['demographic_text'][0]
        x_diag_text = item['text_info']['diagnosis_text'][0]
        x_treatment_text = item['text_info']['treatment_text'][0]
        x_cancer_text = item['text_info']['cancer_text'][0]
        omic_missing = item['omic_missing']
        
        h_path_bag = self.wsi_net(x_path).unsqueeze(1)
        ### Omic encoding
        if not omic_missing:
            ### Omic encoding
            h_omic = [self.sig_networks[idx].forward(sig_feat.unsqueeze(1), mask=omic_mask) for idx, (sig_feat, omic_mask) in enumerate(zip(x_omic, x_omic_mask))]
            h_omic_bag = torch.stack(h_omic).unsqueeze(1) ### omic embeddings are stacked (to be used in co-attention)
        else:
            h_omic_bag = torch.full((6, 1, self.embed_size), fill_value=1e-4).cuda()
    
        ### Text encoding
        h_cancer_text = self.text_encoder(x_cancer_text)
        h_demo_text = self.text_encoder(x_demo_text)
        h_diag_text = self.text_encoder(x_diag_text)
        h_treatment_text = self.text_encoder(x_treatment_text)
        h_text_bag = torch.stack([h_cancer_text, h_demo_text, h_diag_text, h_treatment_text], dim=0)      # 4x1xC
        
        ### Coattn
        A_path_coattn, _ = self.path_coattn(h_path_bag, h_text_bag)
        h_path_coattn = torch.mm(A_path_coattn.squeeze(), h_path_bag.squeeze()).unsqueeze(1)     # 4x1xC
        A_omic_coattn, _ = self.omic_coattn(h_omic_bag, h_text_bag)
        h_omic_coattn = torch.mm(A_omic_coattn.squeeze(), h_omic_bag.squeeze()).unsqueeze(1)     # 4x1xC
        
        ### Path
        h_path_trans = self.path_transformer(h_text_bag, h_path_coattn)             # 4x1xC
        A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
        h_path_res = self.path_rho(h_path).unsqueeze(1)

        ### Omic
        h_omic_trans = self.omic_transformer(h_text_bag, h_omic_coattn)            # 4x1xC
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)                  # 1x4
        h_omic = torch.mm(F.softmax(A_omic, dim=1) , h_omic)    # 1xC
        h_omic_res = self.omic_rho(h_omic).unsqueeze(1)             # 1x1xC
        
        h_res = self.mm(h_omic_res, h_path_res, h_text_bag)
        
        
        ### Get cancer type and diagnosis results
        dataset_name = item['data_belong']
        disease_id = torch.tensor([self.disease_to_id[dataset_name[0]]]).cuda()
        routing_diag_text = item['text_info']['diagnosis_text'][0].split('at')[0].split('has')[-1]
        disease_emb = self.disease_embedding(disease_id)
        tumor_emb = self.text_encoder(routing_diag_text)
        routing_feature = torch.cat([disease_emb,tumor_emb],dim=-1)
        
        logits = torch.stack([expert(h_path_res=h_path_res, h_omic_res=h_omic_res, h_text_bag=h_text_bag)['logits'].squeeze() for expert in self.experts], dim=0).unsqueeze(1)  # (num_experts, batch_size, d_model)
        combine_scores = torch.softmax(self.combine_weight @ routing_feature.T, dim=1)  # (batch_size, num_experts)
        combined_h = torch.einsum('ij,jik->ik', combine_scores, logits)  # (batch_size, d_model)
        
        return combined_h.mean(dim=0), h_res.squeeze(), h_omic_res.squeeze(), h_path_res.squeeze()


#############################
### UMPSNet Implementation ###
#############################
class UMPSNet(nn.Module):
    def __init__(self, cfg,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25,ot_reg=0.1, ot_tau=0.5, ot_impl="pot-uot-l2"):
        super(UMPSNet, self).__init__()
        self.fusion = cfg.model_fusion
        self.omic_sizes = cfg.model_omic_sizes
        self.n_classes = cfg.model_num_classes
        self.size_dict_WSI = {"small": [768, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        size = self.size_dict_WSI[model_size_wsi]
        self.moe = MixtureOfExperts(n_classes=self.n_classes, size=size, num_experts=cfg.experts_num, omic_sizes=self.omic_sizes,ot_reg=ot_reg, ot_tau=ot_tau, 
                                    ot_impl=ot_impl, dropout=dropout, d_model=256, nhead=8, dropout_decoder=0.1)
        
        ### Dataset classifier
        self.classifier_dataset_h = nn.Sequential(*[nn.Linear(size[2], size[2]// 2), nn.ReLU(), nn.Dropout(0.5), nn.Linear(size[2]//2, 5)])
        ### Classifier
        self.classifier = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(0.5), nn.Linear(size[2], self.n_classes)])

    def forward(self, item):    
        h, h_res, h_omic, h_path = self.moe(item)
        
        logits = h.unsqueeze(0)
        h_omic = h_omic.unsqueeze(0)
        h_path = h_path.unsqueeze(0)
        
        ### Dataset Layer
        logits_dataset_h =self.classifier_dataset_h(h_res.unsqueeze(0))
        
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        output ={
            'logits_dataset_h': logits_dataset_h,
            'hazards': hazards, 'S':S,
        }
        return output