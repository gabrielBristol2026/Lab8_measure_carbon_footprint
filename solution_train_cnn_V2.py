import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_convs: int):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGGMNIST(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            VGGBlock(1, 32, 2),
            VGGBlock(32, 64, 2),
            VGGBlock(64, 128, 2),
        )
        # (TODO) Increase FC from 256 to 512
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformerMNIST(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 16,
        num_heads: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


MODEL_REGISTRY = {
    "cnn": SimpleCNN,
    "vgg": VGGMNIST,
    "vit": VisionTransformerMNIST,
}


def build_model(model_name: str) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {', '.join(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]()


def train(model, device, train_loader, optimizer, epoch, log_file):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            log_message = (
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            print(log_message)
            log_file.write(log_message + "\n")


@torch.no_grad()
def test(model, device, test_loader, log_file):
    model.eval()
    test_loss = 0.0
    correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        test_loss += F.cross_entropy(logits, target, reduction="sum").item()
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    log_message = (
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    print(log_message)
    log_file.write(log_message + "\n")
    return test_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST training with selectable models")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="number of training epochs")
    parser.add_argument("--log-dir", type=str, default=".", help="directory to save logs")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=list(MODEL_REGISTRY.keys()),
        help="model architecture to train",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1000, shuffle=False)

    model = build_model(args.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.log_dir, exist_ok=True)

    terminal_log_path = os.path.join(args.log_dir, f"terminal_output_lr_{args.lr}.log")
    csv_log_path = os.path.join(args.log_dir, f"log_lr_{args.lr}.csv")

    with open(terminal_log_path, "w") as terminal_log_file:
        terminal_log_file.write(
            f"Model: {args.model}\nLearning rate: {args.lr}\nEpochs: {args.epochs}\n\n"
        )

        with open(csv_log_path, "w") as csv_log_file:
            csv_log_file.write("epoch,test_loss,accuracy\n")

        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, epoch, terminal_log_file)
            test_loss, accuracy = test(model, device, test_loader, terminal_log_file)
            with open(csv_log_path, "a") as csv_log_file:
                csv_log_file.write(f"{epoch},{test_loss},{accuracy}\n")


if __name__ == "__main__":
    main()
