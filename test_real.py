import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(arr, min_val, max_val):
    return (arr - min_val) / (max_val - min_val)


def get_dataset(fileName):
    df_GM = pd.read_excel("./real_P_wave_Record.xlsx").values
    df = pd.read_excel(fileName)

    GM = pd.to_numeric(df["GM"], errors="raise").astype(np.int64).to_numpy()
    St = pd.to_numeric(df["St"], errors="raise").astype(np.int64).to_numpy() - 1

    Bh = pd.to_numeric(df["BuildingHeight"], errors="raise").astype(np.float32).to_numpy()
    Ns = pd.to_numeric(df["NumStories"], errors="raise").astype(np.float32).to_numpy()
    Sh = pd.to_numeric(df["StoryHeight"], errors="raise").astype(np.float32).to_numpy()
    Pa = pd.to_numeric(df["PlanArea"], errors="raise").astype(np.float32).to_numpy()
    Bw = pd.to_numeric(df["Width"], errors="raise").astype(np.float32).to_numpy()
    Bl = pd.to_numeric(df["Length"], errors="raise").astype(np.float32).to_numpy()
    Blw = pd.to_numeric(df["Wl"], errors="raise").astype(np.float32).to_numpy()

    Ci = pd.to_numeric(df["SiteClass"], errors="raise").astype(np.int64).to_numpy() - 1
    Di = pd.to_numeric(df["DesignIntensity"], errors="raise").astype(np.int64).to_numpy() - 6

    PTFA = np.log10(pd.to_numeric(df["PTFA"], errors="raise").to_numpy())

    print("St dtype:", St.dtype)
    print("St unique:", np.unique(St))

    St_onehot = F.one_hot(torch.tensor(St, dtype=torch.long), num_classes=4).float()
    Ci_onehot = F.one_hot(torch.tensor(Ci, dtype=torch.long), num_classes=4).float()
    Di_onehot = F.one_hot(torch.tensor(Di, dtype=torch.long), num_classes=4).float()

    Bh_norm = normalize(Bh, 3.3, 487).reshape(-1, 1)
    Ns_norm = normalize(Ns, 1, 103).reshape(-1, 1)
    Pa_norm = normalize(Pa, 8, 6000).reshape(-1, 1)
    Blw_norm = normalize(Blw, 0, 5).reshape(-1, 1)

    StrInfo = np.hstack((
        St_onehot.numpy(),
        Ci_onehot.numpy(),
        Di_onehot.numpy(),
        Bh_norm,
        Ns_norm,
        Pa_norm,
        Blw_norm
    ))

    X_GM = df_GM[GM][:, :300]
    X_str = StrInfo
    Y = PTFA

    print("Shape X_str    :", X_str.shape)
    print("Shape X_GM     :", X_GM.shape)
    print("Shape Y        :", Y.shape)

    return X_str, X_GM, Y


class Dataset:
    def __init__(self, test_path):
        self.test_path = test_path

    def load(self):
        X_str_test, X_GM_test, Y_test = get_dataset(self.test_path)

        self.X_str_test = X_str_test
        self.X_GM_test = X_GM_test
        self.Y_test = Y_test


class StrModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )

    def forward(self, x):
        return self.model(x)


class StructTokenizer(nn.Module):
    def __init__(self, input_dim=16, token_dim=16):
        super().__init__()

        self.embeds = nn.ModuleList([
            nn.Linear(1, token_dim) for _ in range(input_dim)
        ])

        self.pos_emb = nn.Parameter(torch.randn(1, input_dim, token_dim))

    def forward(self, x):
        tokens = []
        for i in range(x.shape[1]):
            xi = x[:, i:i+1]
            tokens.append(self.embeds[i](xi))

        tokens = torch.stack(tokens, dim=1)
        tokens = tokens + self.pos_emb

        return tokens


class StructSelfAttention(nn.Module):
    def __init__(self, dim=16, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out + x


class GMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=3, out_dim=16):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)

        x = x.unsqueeze(-1)   # (B,T,1)

        out, _ = self.lstm(x)
        out = self.fc(out)

        return out   # (B,T,16)


class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, dim=16, num_heads=4):
        super().__init__()

        self.attn_gm2s = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.attn_s2gm = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, gm_feat, struct_tokens):

        # GM → Struct
        gm2s, attn_gm2s = self.attn_gm2s(
            gm_feat, struct_tokens, struct_tokens
        )
        gm2s = gm2s + gm_feat

        # Struct → GM
        s2gm, attn_s2gm = self.attn_s2gm(
            struct_tokens, gm_feat, gm_feat
        )
        s2gm = s2gm + struct_tokens

        return gm2s, s2gm, attn_gm2s, attn_s2gm



class DualAttentionPooling(nn.Module):
    def __init__(self, dim=16):
        super().__init__()

        self.time_score = nn.Linear(dim, 1)
        self.struct_score = nn.Linear(dim, 1)

        self.fusion = nn.Linear(dim * 2, dim)

    def forward(self, gm2s, s2gm):

        w_t = torch.softmax(self.time_score(gm2s), dim=1)
        gm_vec = torch.sum(w_t * gm2s, dim=1)

        w_s = torch.softmax(self.struct_score(s2gm), dim=1)
        struct_vec = torch.sum(w_s * s2gm, dim=1)

        fused = torch.cat([gm_vec, struct_vec], dim=1)
        fused = self.fusion(fused)

        return fused, w_t, w_s


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

        var = F.softplus(log_var) + 1e-6

        return mu, var


class Model_NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.str_model = StrModel()
        self.gm_model = GMModel()

        self.tokenizer = StructTokenizer()
        self.struct_attn = StructSelfAttention()

        self.cross_attn = BiDirectionalCrossAttention()

        self.pool = DualAttentionPooling()

        self.out_model = OutModel(input_dim=32)

    def forward(self, str_input, gm_input):

        str_feat = self.str_model(str_input)              # (B,16)
        struct_tokens = self.tokenizer(str_feat)          # (B,16,16)
        struct_tokens = self.struct_attn(struct_tokens)

        gm_feat = self.gm_model(gm_input)                 # (B,T,16)

        gm2s, s2gm, attn_gm2s, attn_s2gm = self.cross_attn(
            gm_feat, struct_tokens
        )

        cross_vec, w_t, w_s = self.pool(gm2s, s2gm)

        gm_global = torch.mean(gm_feat, dim=1)

        x = torch.cat([gm_global, cross_vec], dim=1)

        mu, var = self.out_model(x)

        return mu, var, attn_gm2s, attn_s2gm


def predict(model, dataset, device, batch_size=1):
    model.eval()

    X_str = torch.tensor(dataset.X_str_test, dtype=torch.float32)
    X_GM  = torch.tensor(dataset.X_GM_test, dtype=torch.float32)
    Y     = torch.tensor(dataset.Y_test, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_str, X_GM, Y),
        batch_size=batch_size,
        shuffle=False
    )

    mu_list = []
    var_list = []
    y_list = []

    with torch.no_grad():
        for str_input, gm_input, y in loader:
            str_input = str_input.to(device)
            gm_input = gm_input.to(device)

            mu, var, _, _ = model(str_input, gm_input)

            mu_list.append(mu.cpu())
            var_list.append(var.cpu())
            y_list.append(y.cpu())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    mu = torch.cat(mu_list, dim=0).numpy()
    var = torch.cat(var_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy()

    return mu, var, y


if __name__ == "__main__":

    batch_size = 1
    N_BOOTSTRAP = 30

    test_path = "./real_str_data.xlsx"

    model_dir = "./model"

    output_root = "./real_result_data"
    os.makedirs(output_root, exist_ok=True)

    start = time.time()

    for b in range(1, N_BOOTSTRAP + 1):
        print(f"\n=========== Testing Bootstrap Model {b} ===========")

        dataset = Dataset(test_path=test_path)
        dataset.load()

        model = Model_NN().to(device)

        model_path = os.path.join(model_dir, f"bootstrap_{b}.pth")

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        save_dir = os.path.join(output_root, f"bootstrap_{b}")
        os.makedirs(save_dir, exist_ok=True)

        mu_test, var_test, y_test = predict(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=batch_size
        )

        np.savetxt(os.path.join(save_dir, "mu_real.txt"), mu_test)
        np.savetxt(os.path.join(save_dir, "var_real.txt"), var_test)
        np.savetxt(os.path.join(save_dir, "y_real.txt"), y_test)

        print(f"Bootstrap {b} test prediction saved to: {save_dir}")

    end = time.time()
    print("Total time =", end - start)