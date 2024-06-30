import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 画像を処理するための変換を定義
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature = resnet18()
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h


def predict(content):
    # PIL形式で画像をロードし、RGBに変換
    image = content.convert("RGB")
    # 画像を前処理
    image = transform(image).to(device)
    # バッチ次元を追加（torch.nnはミニバッチを想定しているため）
    image = image.unsqueeze(0)
    # 保存されたモデルをCPU上でロード
    net = Net().cpu().eval()
    net.load_state_dict(
        torch.load("./model/model_resnet18_state.pth", map_location=torch.device("cpu"))
    )
    # 予測を実行
    with torch.no_grad():  # 勾配計算を無効化
        output = net(image)
    # 出力から予測結果を得る（例：最もスコアが高いクラス）
    predicted = torch.argmax(output, 1)

    return predicted.item()
