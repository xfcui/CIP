import torch
import torch.nn as nn
from CIP import layers, losses
import json
import os
import dgl
import dgl.function as dglfn
import dgl.nn.pytorch as dglnn
class GNNs(nn.Module):
    def __init__(self, nLigNode, nLigEdge, nLayer, nHid, JK, GNN):
        super(GNNs, self).__init__()
        if GNN == 'GCN':
            self.Encoder = layers.GCN(nLigNode, hidden_feats=[nHid] * nLayer)
            
        elif GNN == 'GAT':
            self.Encoder = layers.GAT(nLigNode, hidden_feats=[nHid] * nLayer)
          
        elif GNN == 'GIN':
            self.Encoder = layers.GIN(nLigNode, nHid, nLayer, num_mlp_layers=2, dropout=0.1, learn_eps=False,
                               neighbor_pooling_type='sum', JK=JK)
          
        elif GNN == 'EGNN':
            self.Encoder = layers.EGNN(nLigNode, nLigEdge, nHid, nLayer, dropout=0.1, JK=JK)
         
        elif GNN == 'AttentiveFP':
            self.Encoder = layers.ModifiedAttentiveFPGNNV2(120, nLigEdge, nLayer, nHid, 0.1, JK)
          

    def forward(self, Graph, Perturb=None):
        Node_Rep = self.Encoder(Graph, Perturb)
        return Node_Rep

class ASRP_head(nn.Module):
    def __init__(self, config):
        super(ASRP_head, self).__init__()

        self.readout = layers.ReadsOutLayer(config.model.inter_out_dim, config.model.readout)
        self.FC = layers.FC(config.model.inter_out_dim * 2, config.model.fintune_fc_hidden_dim, config.model.dropout, config.model.out_dim)
        self.regression_loss_fn = nn.MSELoss(reduce=False)
        self.ranking_loss_fn = losses.pairwise_BCE_loss(config)


    def forward(self, bg_inter, bond_feats_inter, ass_des, labels, select_flag):
        graph_embedding = self.readout(bg_inter, bond_feats_inter)
        affinity_pred = self.FC(graph_embedding)
        y_pred_num = len(affinity_pred)
        assert y_pred_num % 2 == 0
        regression_loss = self.regression_loss_fn(affinity_pred[:y_pred_num // 2], labels[:y_pred_num // 2])
        labels_select = labels[:y_pred_num // 2][select_flag[:y_pred_num // 2]]
        affinity_pred_select = affinity_pred[:y_pred_num // 2][select_flag[:y_pred_num // 2]]
        regression_loss_select = regression_loss[select_flag[:y_pred_num // 2]].sum()

        ranking_loss, relation, relation_pred = self.ranking_loss_fn(graph_embedding, labels)
        ranking_loss_select = ranking_loss[select_flag[:y_pred_num // 2]].sum()
        relation_select = relation[select_flag[:y_pred_num // 2]]
        relation_pred_selcet = relation_pred[select_flag[:y_pred_num // 2]]

        return regression_loss_select, ranking_loss_select,\
               labels_select, affinity_pred_select,\
               relation_select, relation_pred_selcet

    def inference(self, bg_inter, bond_feats_inter, ass_des, labels, select_flag): 
        graph_embedding = self.readout(bg_inter, bond_feats_inter)
        affinity_pred = self.FC(graph_embedding)

        regression_loss = self.regression_loss_fn(affinity_pred, labels)  
        regression_loss_select = regression_loss[select_flag].sum()


        labels_select = labels[select_flag]
        affinity_pred_select = affinity_pred[select_flag]

        return regression_loss_select, labels_select, affinity_pred_select

class Affinity_GNNs_MTL(nn.Module):
    def __init__(self, config):
        super(Affinity_GNNs_MTL, self).__init__()

        lig_node_dim = config.model.lig_node_dim
        lig_edge_dim = config.model.lig_edge_dim
        pro_node_dim = config.model.pro_node_dim
        pro_edge_dim = config.model.pro_edge_dim
        layer_num = config.model.num_layers
        hidden_dim = config.model.hidden_dim
        jk = config.model.jk
        GNN = config.model.GNN_type

        self.lig_encoder = GNNs(lig_node_dim, lig_edge_dim, layer_num, hidden_dim, jk, GNN)
        self.pro_encoder = GNNs(pro_node_dim, pro_edge_dim, layer_num, hidden_dim, jk, GNN)

        self.cross_attention = CrossAttention(128, 128)

        if config.model.jk == 'concat':
            self.noncov_graph = layers.DTIConvGraph3Layer(hidden_dim * (layer_num + layer_num) + config.model.inter_edge_dim, config.model.inter_out_dim, config.model.dropout)
        else:
            self.noncov_graph = layers.DTIConvGraph3Layer(hidden_dim * 2 + config.model.inter_edge_dim, config.model.inter_out_dim, config.model.dropout)
        self.softmax = nn.Softmax(dim=1)

        self.IC50_ASRP_head = ASRP_head(config)
        self.K_ASRP_head = ASRP_head(config)

        self.a_init = nn.Linear(21, 120)
        self.b_init = nn.Linear(18, 120)
        self.r_mt = MT(120)
        self.a_mt = MT(120)

        self.sum_pool = dglnn.SumPooling()
        self.config = config

    def multi_head_inference(self, bg_inter, bond_feats_inter, labels, ass_des, IC50_f, K_f):

        regression_loss_K, affinity_K, affinity_pred_K = \
            self.K_ASRP_head.inference(bg_inter, bond_feats_inter, ass_des, labels, K_f) 
        regression_loss_IC50, affinity_IC50, affinity_pred_IC50 = \
            self.IC50_ASRP_head.inference(bg_inter, bond_feats_inter, ass_des, labels, IC50_f) 
        return (regression_loss_IC50, regression_loss_K),\
               (affinity_pred_IC50, affinity_pred_K), \
               (affinity_IC50, affinity_K)

    def multi_head_asrp(self, bg_inter, bond_feats_inter, labels, ass_des, IC50_f, K_f):
        regression_loss_IC50, ranking_loss_IC50, \
        affinity_IC50, affinity_pred_IC50, \
        relation_IC50, relation_pred_IC50 = self.IC50_ASRP_head(bg_inter, bond_feats_inter, ass_des, labels, IC50_f)

        regression_loss_K, ranking_loss_K, \
        affinity_K, affinity_pred_K, \
        relation_K, relation_pred_K = self.K_ASRP_head(bg_inter, bond_feats_inter, ass_des, labels, K_f)

        return (regression_loss_IC50, regression_loss_K),\
               (ranking_loss_IC50, ranking_loss_K), \
               (affinity_pred_IC50, affinity_pred_K), \
               (relation_pred_IC50, relation_pred_K), \
               (affinity_IC50, affinity_K), \
               (relation_IC50, relation_K)

    def alignfeature(self,bg_lig,bg_prot,node_feats_lig,node_feats_prot):
        inter_feature = torch.cat((node_feats_lig,node_feats_prot))
        lig_num,prot_num = bg_lig.batch_num_nodes(),bg_prot.batch_num_nodes()
        lig_start, prot_start = lig_num.cumsum(0) - lig_num, prot_num.cumsum(0) - prot_num
        inter_start = lig_start + prot_start
        for i in range(lig_num.shape[0]):
            inter_feature[inter_start[i]:inter_start[i]+lig_num[i]] = node_feats_lig[lig_start[i]:lig_start[i]+lig_num[i]]
            inter_feature[inter_start[i]+lig_num[i]:inter_start[i]+lig_num[i]+prot_num[i]] = node_feats_prot[prot_start[i]:prot_start[i]+prot_num[i]]
        return inter_feature
    
    
    def forward(self, batch, ASRP=True):

        bg_lig, bg_prot, bg_inter, labels, _, ass_des, IC50_f, K_f = batch

        va_init = self.b_init(bg_lig.ndata['h'].float())  
        vr_init = self.a_init(bg_prot.ndata['h'].float())   

        bg_lig.ndata['h'] = self.a_mt(bg_prot, vr_init, bg_lig, va_init)          
        bg_prot.ndata['h'] = self.r_mt(bg_lig, va_init, bg_prot, vr_init)        

        node_feats_lig = self.lig_encoder(bg_lig)
        node_feats_prot = self.pro_encoder(bg_prot)

        inter_feature= self.alignfeature(bg_lig, bg_prot, node_feats_lig, node_feats_prot)
            
        bg_inter.ndata['h'] = inter_feature
       
        bond_feats_inter = self.noncov_graph(bg_inter)

        if ASRP:
            return self.multi_head_asrp(bg_inter, bond_feats_inter, labels, ass_des, IC50_f, K_f)
        else:
            return self.multi_head_inference(bg_inter, bond_feats_inter, labels, ass_des, IC50_f, K_f)
        

class MT(nn.Module):

    def __init__(self, in_dim):  
        super().__init__()

        self.A = nn.Linear(in_dim, 64)  
        self.B = nn.Linear(in_dim, 8)
        self.C = nn.Linear(64, in_dim)
        self.sum_pool = dglnn.SumPooling()
        self.D = nn.Linear(in_dim, 120)           
        self.E = nn.Linear(in_dim, 120)
        # self.batchnormD = nn.BatchNorm1d(120)
        

    def forward(self, ga, va, gb, vb):   
        s = self.A(va)   
        h = self.B(va) 
        with ga.local_scope():    
            ga.ndata['h'] = h
            h = dgl.softmax_nodes(ga, 'h')  
            ga.ndata['h'] = h
            ga.ndata['s'] = s

            gga = dgl.unbatch(ga)  
            gp_ = torch.stack([torch.mm(g.ndata['s'].T, g.ndata['h']) for g in gga])   
            gp_ = self.C(gp_.mean(dim=-1)) 

        # gp2_ = self.batchnormD(self.D(gp_)) 
        gp2_ = self.D(gp_)
        gp3_ = dgl.broadcast_nodes(gb, gp2_) 
        gp3_ = gp3_.permute(1,0)           
        r_ = torch.sum(torch.mm(self.E(vb),gp3_),dim=-1)
        pad_ = torch.sigmoid(r_)
        vbb = vb + vb * pad_.unsqueeze(1)

        return vbb

class CrossAttention(nn.Module):
    def __init__(self, node_feat_size, attention_size, dropout_rate=0.2):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(node_feat_size, attention_size)
        self.key = nn.Linear(node_feat_size, attention_size)
        self.value = nn.Linear(node_feat_size, attention_size)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = torch.sqrt(torch.tensor(attention_size, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.BatchNorm1d(attention_size)
        self.self_attention_norm = nn.LayerNorm(attention_size)
        self.ffn = FeedForwardNetwork(128, 128)
        
        self.gru = nn.GRU(attention_size, 128, batch_first=True)

        self.fc = nn.Linear(128, attention_size)

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, ligand_feats, protein_feats):

        ligand_feats = self.self_attention_norm(ligand_feats)
        protein_feats = self.self_attention_norm(protein_feats)

        query = ligand_feats
        key = protein_feats
        value = protein_feats
        attention_weights = self.softmax(torch.matmul(query, key.transpose(-2, -1)) / self.scale)
       
        attended_feats = torch.matmul(attention_weights, value)
        attended_feats = self.dropout(attended_feats)

        attended_feats = attended_feats + ligand_feats
        attended_feats = self.layer_norm(attended_feats)

        attended_feats= self.ffn(attended_feats)
        attended_feats = self.dropout(attended_feats)
        attended_feats = attended_feats + ligand_feats

        attended_feats, _ = self.gru(attended_feats.unsqueeze(0))
        attended_feats = self.fc(attended_feats.squeeze(0))
        attended_feats = attended_feats + ligand_feats

  
        return self.alpha * attended_feats + (1 - self.alpha) * ligand_feats
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x
 