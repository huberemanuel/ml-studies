import torch
import pdb
import pandas as pd
import tqdm
import numpy as np
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class MyDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.from_numpy(x).to(torch.float)
        self.y = torch.from_numpy(y).to(torch.float)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class MyNN(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.net = nn.Linear(input_features, 1, dtype=torch.float)

    def forward(self, x):
        return self.net(x)


class MyTrainer:
    def __init__(self, model, l: float = 0.003, l1: bool = False):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.writer = SummaryWriter()
        self.l1 = l1
        self.l = l

    def fit(self, train_loader, epoch):
        losses = []
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.cuda(), b_y.cuda()
            y_pred = self.model(b_x).squeeze(dim=-1)
            loss = self.loss_fn(y_pred, b_y)
            if self.l1:
                loss = loss + self.l * parameters_to_vector(
                    self.model.parameters()
                ).norm(1)
            else:
                loss = loss + self.l * parameters_to_vector(
                    self.model.parameters()
                ).norm(2)
            losses.append(loss.item())

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        losses = np.mean(losses)
        self.writer.add_scalar("loss/train", losses, epoch)

    def eval(self, eval_loader, epoch):
        model.eval()
        losses = []
        y_preds = []
        y_trues = []
        with torch.no_grad():
            for b_x, b_y in eval_loader:
                b_x, b_y = b_x.cuda(), b_y.cuda()
                y_pred = self.model(b_x).squeeze(dim=-1)
                loss = self.loss_fn(y_pred, b_y)
                if self.l1:
                    loss = loss + self.l * parameters_to_vector(
                        self.model.parameters()
                    ).norm(1)
                else:
                    loss = loss + self.l * parameters_to_vector(
                        self.model.parameters()
                    ).norm(2)
                y_preds.extend((torch.sigmoid(y_pred) > 0.5).cpu())
                y_trues.extend(b_y.cpu())

                losses.append(loss.item())
        losses = np.mean(losses)
        acc = accuracy_score(y_trues, y_preds)
        self.writer.add_scalar("loss/val", losses, epoch)
        self.writer.add_scalar("acc/val", acc, epoch)

    def flush(self):
        self.writer.flush()


if __name__ == "__main__":
    epochs = 40
    batch_size = 128
    learning_rate = 0.00003
    l = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # https://www.kaggle.com/competitions/bnp-paribas-cardif-claims-management/data
    train_df = pd.read_csv("data/paribas_train.csv")
    train_df = train_df.sample(10000)
    y = train_df.pop("target").values
    train_df = train_df.select_dtypes([np.number])
    train_df = train_df.fillna(value=0)
    x = pd.get_dummies(train_df).values
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=42
    )

    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    X_train = scaler1.fit_transform(X_train)
    X_test = scaler1.transform(X_test)
    # y_train = scaler2.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    # y_test = scaler2.transform(y_test.reshape(-1, 1)).reshape(-1)

    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    samples, labels = next(iter(train_dataloader))

    model = MyNN(x.shape[1]).to(device)

    trainer = MyTrainer(model, l, True)

    for epoch in tqdm.tqdm(range(epochs)):
        trainer.fit(train_dataloader, epoch)
        trainer.eval(test_dataloader, epoch)
    trainer.flush()

    model.cpu()
    print("L1 model: ")
    for name, param in model.named_parameters():
        print(name, param)
    print("Norm l1: ", parameters_to_vector(model.parameters()).norm(1))
    print("Norm l2: ", parameters_to_vector(model.parameters()).norm(2))

    v = parameters_to_vector(model.parameters())
    print(f"{v[v.abs() < 1e-5].size()} would be removed")

    model = MyNN(x.shape[1]).to(device)

    trainer = MyTrainer(model, l, False)

    for epoch in tqdm.tqdm(range(epochs)):
        trainer.fit(train_dataloader, epoch)
        trainer.eval(test_dataloader, epoch)
    trainer.flush()

    model.cpu()
    print("L2 model: ")
    for name, param in model.named_parameters():
        print(name, param)
    print("Norm l1: ", parameters_to_vector(model.parameters()).norm(1))
    print("Norm l2: ", parameters_to_vector(model.parameters()).norm(2))

    v = parameters_to_vector(model.parameters())
    print(f"{v[v.abs() < 1e-5].size()} would be removed")
