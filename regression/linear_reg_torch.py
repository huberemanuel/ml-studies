import torch
import tqdm
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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
    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        self.writer = SummaryWriter()

    def fit(self, train_loader, epoch):
        losses = []
        for batch, (b_x, b_y) in enumerate(train_loader):
            b_x, b_y = b_x.cuda(), b_y.cuda()
            y_pred = self.model(b_x).squeeze(dim=-1)
            loss = self.loss_fn(y_pred, b_y)
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
        rmse = []
        with torch.no_grad():
            for batch, (b_x, b_y) in enumerate(eval_loader):
                b_x, b_y = b_x.cuda(), b_y.cuda()
                y_pred = self.model(b_x).squeeze(dim=-1)
                loss = self.loss_fn(y_pred, b_y)
                rmse.append(mean_squared_error(b_y.cpu(), y_pred.cpu(), squared=False))

                losses.append(loss.item())
        losses = np.mean(losses)
        self.writer.add_scalar("loss/val", losses, epoch)
        self.writer.add_scalar("rmse/val", losses, epoch)
        rmse = np.mean(rmse)

    def flush(self):
        self.writer.flush()


if __name__ == "__main__":
    epochs = 100
    batch_size = 32
    learning_rate = 0.003
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    data = load_diabetes()
    x = data["data"]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=42
    )

    def toFloat32(x):
        return x.to(torch.float)

    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    samples, labels = next(iter(train_dataloader))

    model = MyNN(x.shape[1]).to(device)

    trainer = MyTrainer(model)

    for epoch in tqdm.tqdm(range(epochs)):
        trainer.fit(train_dataloader, epoch)
        trainer.eval(test_dataloader, epoch)
    trainer.flush()
