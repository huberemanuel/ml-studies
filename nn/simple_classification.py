import torch
import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader


from sklearn.metrics import accuracy_score


class MyNN(nn.Module):
    def __init__(self, w, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(w * h, 16), nn.ReLU(), nn.Linear(16, 10)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Trainer:
    def __init__(self, model: nn.Module, lr: float, device: str, writer):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr)
        self.device = device
        self.writer = writer

    def fit(self, train_loader, epoch):
        losses = []
        model.train()
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(self.device), b_y.to(self.device)
            y_pred = self.model(b_x)
            loss = self.loss_fn(y_pred, b_y)
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        losses = sum(losses) / len(losses)
        # print(f"Train loss on {epoch}: {losses}")
        self.writer.add_scalar("loss/train", losses, epoch)

    def eval(self, test_loader, epoch):
        losses = []
        y_trues = []
        y_preds = []
        self.model.eval()
        with torch.no_grad():
            for b_x, b_y in test_loader:
                b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                y_pred = self.model(b_x)
                y_cls = y_pred.argmax(-1).cpu().numpy()
                y_preds.extend(y_cls)
                y_trues.extend(b_y.cpu())
                loss = self.loss_fn(y_pred, b_y)
                losses.append(loss.item())
        losses = sum(losses) / len(losses)
        acc = accuracy_score(y_trues, y_preds)
        # print(f"Val loss on {epoch}: {losses}")
        self.writer.add_scalar("loss/val", losses, epoch)
        self.writer.add_scalar("acc/val", acc, epoch)


if __name__ == "__main__":
    epochs = 10
    batch_size = 32
    learning_rate = 0.003
    writer = SummaryWriter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    train_loader = DataLoader(training_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    w, h = training_data[0][0].size()[1:]
    model = MyNN(w, h).to(device)
    print(model)

    trainer = Trainer(model, learning_rate, device, writer)
    for epoch in tqdm.tqdm(range(epochs)):
        trainer.fit(train_loader, epoch)
        trainer.eval(test_loader, epoch)
