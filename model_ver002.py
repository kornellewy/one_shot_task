"""
source:
https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
https://towardsdatascience.com/building-a-one-shot-learning-network-with-pytorch-d1c3a5fafa4a
https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
"""
import torch
import torch.nn as nn

from model_ver001 import FashionCNNVer001

class FashionCNNVer002(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        self.base_model.fc1 = nn.Identity()
        self.base_model.Dropout2d = nn.Identity()
        self.base_model.fc2 = nn.Identity()
        self.base_model.fc3 = nn.Identity()

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, l1, l2, x):
        # l1 - 0 - flats
        # l2 - 1 - heels
        l1 = self.base_model(l1)
        l1 = self.sigmoid(l1)

        l2 = self.base_model(l2)
        l2 = self.sigmoid(l2)

        x = self.base_model(x)
        x = self.sigmoid(x)

        cos_l1 = self.cos(x, l1)
        cos_l2 = self.cos(x, l2)

        if cos_l1 > cos_l2:
            return torch.tensor([0])
        else:
            return torch.tensor([1])

        
if __name__ == '__main__':
    device = torch.device("cpu")
    base_model_path = 'models/test_loss_0.005_accuracy_0.9984.pth'
    base_model = torch.load(base_model_path).to(device)
    input_tensor = torch.rand(1, 1, 28, 28)
    model = FashionCNNVer002(base_model=base_model).to(device)
    output_tensor = model.forward(input_tensor, input_tensor, input_tensor)
    print(output_tensor.shape)
    print(output_tensor)
    