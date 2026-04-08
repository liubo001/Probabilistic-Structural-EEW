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
    df_GM = pd.read_csv("../P_Data.csv").values
    df = pd.read_csv(fileName)

    input_data = df.iloc[:, :12].values

    GM = input_data[:, 0].astype(int)
    St = input_data[:, 1] - 1
    Sn = input_data[:, 2]
    Bh = input_data[:, 3]
    Ns = input_data[:, 4]
    Sh = input_data[:, 5]
    Pa = input_data[:, 6]
    Bw = input_data[:, 7]
    Bl = input_data[:, 8]
    Blw = input_data[:, 9]
    Ci = input_data[:, 10] - 1
    Di = input_data[:, 11] - 6

    MPFA = np.log10(df.iloc[:, 15].values * 100)

    St_onehot = torch.nn.functional.one_hot(
        torch.tensor(St, dtype=torch.long), num_classes=4).float()

    Ci_onehot = torch.nn.functional.one_hot(
        torch.tensor(Ci, dtype=torch.long), num_classes=4).float()

    Di_onehot = torch.nn.functional.one_hot(
        torch.tensor(Di, dtype=torch.long), num_classes=4).float()

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

    X_GM = df_GM[GM][:, 5:305]
    X_str = StrInfo
    Y = MPFA

    print("Shape X_str    :", X_str.shape)
    print("Shape X_GM     :", X_GM.shape)
    print("Shape Y        :", Y.shape)

    return X_str, X_GM, Y


class Dataset:
    def __init__(self, train_path, val_path, test_path):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

    def load(self):
        X_str_train, X_GM_train, Y_train = get_dataset(self.train_path)
        X_str_val, X_GM_val, Y_val = get_dataset(self.val_path)
        X_str_test, X_GM_test, Y_test = get_dataset(self.test_path)

        self.X_str_train = X_str_train
        self.X_GM_train = X_GM_train
        self.Y_train = Y_train

        self.X_str_val = X_str_val
        self.X_GM_val = X_GM_val
        self.Y_val = Y_val

        self.X_str_test = X_str_test
        self.X_GM_test = X_GM_test
        self.Y_test = Y_test


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


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


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0

    for str_input, GM_input, targets in train_loader:
        targets = targets.unsqueeze(1)

        str_input = str_input.to(device)
        GM_input = GM_input.to(device)
        y = targets.to(device)

        optimizer.zero_grad()

        mu, var, _, _ = model(str_input, GM_input)

        loss = criterion(mu, y, var)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def val_epoch(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0.0

    with torch.no_grad():
        for str_input, GM_input, targets in val_loader:

            targets = targets.unsqueeze(1)

            str_input = str_input.to(device)
            GM_input = GM_input.to(device)
            y = targets.to(device)

            mu, var, _, _ = model(str_input, GM_input)

            loss = criterion(mu, y, var)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def train_model(model, dataset, batch_size, nb_epoch, device, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    model.to(device)

    str_input_train = torch.tensor(dataset.X_str_train, dtype=torch.float32)
    GM_input_train = torch.tensor(dataset.X_GM_train, dtype=torch.float32)
    Y_train = torch.tensor(dataset.Y_train, dtype=torch.float32)

    str_input_val = torch.tensor(dataset.X_str_val, dtype=torch.float32)
    GM_input_val = torch.tensor(dataset.X_GM_val, dtype=torch.float32)
    Y_val = torch.tensor(dataset.Y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(str_input_train, GM_input_train, Y_train),
        batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(str_input_val, GM_input_val, Y_val),
        batch_size=batch_size, shuffle=False
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.GaussianNLLLoss()

    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

    best_val = float('inf')

    train_losses, val_losses = [], []

    for epoch in range(nb_epoch):

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = val_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{nb_epoch}] "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    np.savetxt(f"{save_dir}/train_losses.txt", train_losses)
    np.savetxt(f"{save_dir}/val_losses.txt", val_losses)

    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss.png")
    plt.close()


def predict(model, dataset, device, mode="val", batch_size=512):
    model.eval()

    if mode == "val":
        X_str = torch.tensor(dataset.X_str_val, dtype=torch.float32)
        X_GM  = torch.tensor(dataset.X_GM_val, dtype=torch.float32)
        Y     = torch.tensor(dataset.Y_val, dtype=torch.float32)

    elif mode == "test":
        X_str = torch.tensor(dataset.X_str_test, dtype=torch.float32)
        X_GM  = torch.tensor(dataset.X_GM_test, dtype=torch.float32)
        Y     = torch.tensor(dataset.Y_test, dtype=torch.float32)

    else:
        raise ValueError("mode must be 'val' or 'test'")

    loader = DataLoader(
        TensorDataset(X_str, X_GM, Y),
        batch_size=batch_size,
        shuffle=False
    )

    mu_list = []
    var_list = []
    Y_list = []

    with torch.no_grad():
        for str_input, gm_input, y in loader:

            str_input = str_input.to(device)
            gm_input  = gm_input.to(device)

            mu, var, _, _ = model(str_input, gm_input)

            mu_list.append(mu.cpu())
            var_list.append(var.cpu())
            Y_list.append(y.cpu())

            torch.cuda.empty_cache()

    mu = torch.cat(mu_list, dim=0).numpy()
    var = torch.cat(var_list, dim=0).numpy()
    Y   = torch.cat(Y_list, dim=0).numpy()

    return mu, var, Y


if __name__=='__main__':
    
    if not os.path.exists("./result_data"):
        os.mkdir("./result_data")
    if not os.path.exists("./model"):
        os.mkdir("./model")

    batch_size, nb_epoch = 1024, 50

    N_BOOTSTRAP = 30

    test_path = "../test_LOEO_LOBO.csv"
    val_path = "../val_LOEO_LOBO.csv"

    all_models = []

    start = time.time()

    for b in range(1, N_BOOTSTRAP + 1):

        print(f"\n=========== Bootstrap Model {b} ===========")

        train_path = f"../train_LOEO_LOBO_{b}.csv"

        dataset = Dataset(train_path=train_path, val_path=val_path, test_path=test_path)
        dataset.load()

        model = Model_NN()

        save_dir = f"./result_data/bootstrap_{b}"
        os.makedirs(save_dir, exist_ok=True)

        train_model(
            model, dataset, batch_size, nb_epoch, device, save_dir
        )

        model.load_state_dict(
            torch.load(f"{save_dir}/best_model.pth")
        )
        model.to(device)

        mu_val, var_val, y_val = predict(model, dataset, device, mode="val")

        np.savetxt(f"{save_dir}/mu_val.txt", mu_val)
        np.savetxt(f"{save_dir}/var_val.txt", var_val)
        np.savetxt(f"{save_dir}/y_val.txt", y_val)


        mu_test, var_test, y_test = predict(model, dataset, device, mode="test")

        np.savetxt(f"{save_dir}/mu_test.txt", mu_test)
        np.savetxt(f"{save_dir}/var_test.txt", var_test)
        np.savetxt(f"{save_dir}/y_test.txt", y_test)


        model_path = f"./model/bootstrap_{b}.pth"
        torch.save(model.state_dict(), model_path)

    end = time.time()
    print("Total time=", end-start)