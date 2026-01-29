import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(values, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in values]

def get_gnn_graph_list(fileName):
    df_GM = pd.read_excel("./real_P_wave_Record.xlsx")
    GMList = df_GM.values.tolist()
    GMList = np.array(GMList)

    df_str = pd.read_excel(fileName)
    df_data_li = df_str.values.tolist()
    
    graph_list = []
    
    for s_li in df_data_li:

        GM_idx = s_li[1]
        
        St = s_li[2]-1

        Ns = int(s_li[6])
        # print(Ns)
        Sh = s_li[7]
        Pa = s_li[8]
        Bw = s_li[9]
        Bl = s_li[10]
        Blw = s_li[11]
        Ci = s_li[12]-1
        Di = s_li[13]-6

        PTFA = np.log10(s_li[16])  # lo

        # --- one-hot ---
        St_tensor = torch.tensor([St], dtype=torch.float).long()
        Ci_tensor = torch.tensor([Ci], dtype=torch.float).long()
        Di_tensor = torch.tensor([Di], dtype=torch.float).long()
        St_onehot = torch.nn.functional.one_hot(St_tensor, num_classes=4).float()
        Ci_onehot = torch.nn.functional.one_hot(Ci_tensor, num_classes=4).float()
        Di_onehot = torch.nn.functional.one_hot(Di_tensor, num_classes=4).float()
        
        Ns_norm = normalize(np.array(Ns).reshape(-1, 1), 1, 103)
        Sh_norm = normalize(np.array(Sh).reshape(-1, 1), 2.2, 5.2)
        Pa_norm = normalize(np.array(Pa).reshape(-1, 1), 8, 6000)
        Blw_norm = normalize(np.array(Blw).reshape(-1, 1), 0, 5)
        node_feat = np.column_stack((St_onehot, Ci_onehot, Di_onehot, Ns_norm, Pa_norm, Blw_norm))

        Sh_norm_val = float(Sh_norm[0][0])

        Nodenum = Ns + 1

        x = np.repeat(node_feat, repeats=Nodenum, axis=0)

        edge_index_list = []
        for i in range(Nodenum - 1):
            edge_index_list.append([i, i + 1])
            edge_index_list.append([i + 1, i])
        edge_index = np.array(edge_index_list, dtype=np.int64).T  # (2,E)

        edge_attr_list = []
        for i in range(Nodenum - 1):
            edge_attr_list.append([Sh_norm_val])   # (1,)
            edge_attr_list.append([Sh_norm_val])

        edge_attr = np.array(edge_attr_list, dtype=np.float32)   # (E,1)

        y = np.array([PTFA], dtype=np.float32)

        GM_array = np.array(GMList[int(GM_idx)][:300] , dtype=np.float32)
        GM_tensor = torch.tensor(GM_array, dtype=torch.float32).unsqueeze(0)   # [1, 300]

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            gm_attr=GM_tensor
        )
        graph_list.append(data)

    print('Shape of dataset X : {}',len(graph_list))

    return graph_list


class Dataset:
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        test_graphs = get_gnn_graph_list(self.filepath)
        self.test_graphs = test_graphs
        print("Number of test graphs:", len(test_graphs))

        print("example:", self.test_graphs[0])
        print("number of nodes:", self.test_graphs[0].num_nodes)
        print("node features:", self.test_graphs[0].x.shape)
        print("label:", self.test_graphs[0].y)


class BuildingGNN(nn.Module):
    def __init__(self, input_dim, hid_dim, edge_dim, gnn_embed_dim, dropout):
        super(BuildingGNN, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hid_dim, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hid_dim, hid_dim*2, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(hid_dim*2, gnn_embed_dim, edge_dim=edge_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.relu(x)
        x, (edge_index_returned, attention_weights) = self.conv3(
            x, edge_index, edge_attr, return_attention_weights=True
        )
        x = self.dropout(x)
        gnn_embedding = global_mean_pool(x, batch)
        return gnn_embedding, edge_index_returned, attention_weights


class GMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=3, out_dim=16, bidirectional=False):
        super(GMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        d = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=bidirectional
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        if x.dim() == 3:       # (B, 1, T)
            x = x.squeeze(1)   # → (B, T)
        x = x.unsqueeze(-1)    # → (B, T, 1)

        out, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]
        out = self.fc(h_last)
        return out


class OutModel(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(64, 1)
        self.log_var_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.net(x)

        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        var = torch.exp(log_var)

        return mu, var


class MultiHeadCrossModalAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, dim_attn, num_heads=4):
        super().__init__()
        assert dim_attn % num_heads == 0, \
            "dim_attn must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        self.q_proj = nn.Linear(dim_q, dim_attn)
        self.k_proj = nn.Linear(dim_kv, dim_attn)
        self.v_proj = nn.Linear(dim_kv, dim_attn)

        self.out_proj = nn.Linear(dim_attn, dim_attn)

        self.scale = self.head_dim ** -0.5

    def forward(self, q, kv):
        B = q.size(0)

        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)

        # Split heads
        Q = Q.view(B, self.num_heads, self.head_dim).unsqueeze(2)
        K = K.view(B, self.num_heads, self.head_dim).unsqueeze(2)
        V = V.view(B, self.num_heads, self.head_dim).unsqueeze(2)

        attn_score = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weight = F.softmax(attn_score, dim=-1)

        out = torch.matmul(attn_weight, V)
        out = out.squeeze(2).reshape(B, -1)
        out = self.out_proj(out)

        return out, attn_weight



class BiDirectionalCrossModalAttention(nn.Module):
    def __init__(self,
                 dim_gm,
                 dim_struct,
                 dim_attn,
                 num_heads=4):
        super().__init__()

        # GM → Structure
        self.attn_gm_to_struct = MultiHeadCrossModalAttention(
            dim_q=dim_gm,
            dim_kv=dim_struct,
            dim_attn=dim_attn,
            num_heads=num_heads
        )

        # Structure → GM
        self.attn_struct_to_gm = MultiHeadCrossModalAttention(
            dim_q=dim_struct,
            dim_kv=dim_gm,
            dim_attn=dim_attn,
            num_heads=num_heads
        )

        self.fusion = nn.Linear(2 * dim_attn, dim_attn)

    def forward(self, gm_feat, struct_feat):
        struct_attn, attn_gm2s = self.attn_gm_to_struct(
            gm_feat, struct_feat
        )

        gm_attn, attn_s2gm = self.attn_struct_to_gm(
            struct_feat, gm_feat
        )

        fused = torch.cat([struct_attn, gm_attn], dim=1)
        fused = self.fusion(fused)

        return fused, attn_gm2s, attn_s2gm



class Model_GNN_NN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim,
                 gnn_embed_dim=16, GM_out_dim=16,
                 attn_dim=16, num_heads=4):
        super().__init__()

        self.gnn_model = BuildingGNN(
            node_in_dim,
            hid_dim=32,
            edge_dim=edge_in_dim,
            gnn_embed_dim=gnn_embed_dim,
            dropout=0.1
        )

        self.GM_model = GMModel(out_dim=GM_out_dim)

        self.cross_attn = BiDirectionalCrossModalAttention(
            dim_gm=GM_out_dim,
            dim_struct=gnn_embed_dim,
            dim_attn=attn_dim,
            num_heads=4
        )

        self.out_model = OutModel(
            input_dim=GM_out_dim + attn_dim
        )

    def forward(self, batch, GM_input):
        gnn_out, _, _ = self.gnn_model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

        gm_out = self.GM_model(GM_input)
        cross_feat, attn_gm2s, attn_s2gm = self.cross_attn(
            gm_out, gnn_out
        )
        x = torch.cat([gm_out, cross_feat], dim=1)
        mu, var = self.out_model(x)

        return mu, var, attn_gm2s, attn_s2gm


    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()


    def evaluate(self, dataset, batch_size):
        self.to(device)
        self.eval()

        test_loader = DataLoader(
            dataset.test_graphs,
            batch_size=batch_size,
            shuffle=False
        )

        all_mu = []
        all_var = []
        all_prob = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch = batch.to(device)

                GM_input = batch.gm_attr.unsqueeze(1).to(device)
                y = batch.y.float().unsqueeze(1).to(device)

                mu, var, _, _ = self(batch, GM_input)

                all_mu.append(mu.cpu().numpy())
                all_var.append(var.cpu().numpy())
                all_targets.append(y.cpu().numpy())

                print(f"Test Batch {batch_idx+1}/{len(test_loader)}")

        mu_all = np.concatenate(all_mu, axis=0).squeeze()
        var_all = np.concatenate(all_var, axis=0).squeeze()
        truths = np.concatenate(all_targets, axis=0).squeeze()

        print("mu shape:", mu_all.shape)
        print("var shape:", var_all.shape)
        print("truth shape:", truths.shape)

        np.savetxt("./predicts-test/mu.txt", mu_all)
        np.savetxt("./predicts-test/var.txt", var_all)
        np.savetxt("./predicts-test/truths.txt", truths)


if __name__ == '__main__':
    
    pred_path = "./predicts-test"
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    if not os.path.exists("./model"):
        print("Please supplement the pre trained model")

    model_name = "best_model"
    batch_size = 1
    filepath   = "./real_str_data.xlsx"

    node_in_dim = 15
    edge_in_dim = 1
    model = Model_GNN_NN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim)

    model_path = './model/%s.pth' % model_name
    model.load_model(model_path)

    dataset = Dataset(filepath=filepath)
    dataset.load()

    start = time.time()
    model.evaluate(dataset, batch_size)

    end = time.time()
    print("Evaluation time:", end - start)