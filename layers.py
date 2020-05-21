import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepOutputLayer(nn.Module):

    def __init__(self, out_dim, y_dim, h_dim, z_dim):
        super(DeepOutputLayer, self).__init__()
        
        self.w_h = nn.Linear(h_dim, y_dim)
        self.w_z = nn.Linear(z_dim, y_dim)
        self.out = nn.Linear(y_dim, out_dim)

    def forward(self, y, h, z):
        h = self.w_h(h)
        z = self.w_z(z)
        return self.out(h + z + y)
