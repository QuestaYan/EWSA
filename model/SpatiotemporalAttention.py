import torch
import torch.nn as nn
import torch.nn.functional as F

def multiplicative(x, data):
    # data = torch.tensor(data)
    # print('attention.py:data.size()',data.size())
    N, D = data.size() #data是(1,256)的tensor
    N, C, L, H, W = x.size()
    # print('attention.py:x.size()',x.size())
    assert D <= C, "data dims must be less than channel dims"
    # print('attention.py:data.view(N, D, 1, 1, 1).size()',data.view(N, D, 1, 1, 1).size())#(1,256,1,1,1)
    # print('attention.py:data.view(N, D, 1, 1, 1).expand(N, D, L, H, W).size()',data.view(N, D, 1, 1, 1).expand(N, D, L, H, W).size())#(1,256,sequence length,H,W)
    x = torch.cat([
        x[:,:D,:,:,:] * data.view(N, D, 1, 1, 1).expand(N, D, L, H, W),
        x[:,D:,:,:,:]
    ], dim=1)
    return x

class AttentiveEncoder(nn.Module):
    """
    Input: (N, 3, L, H, W), (N, D,)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, data_dim, linf_max=0.016, kernel_size=(1,11,11), padding=(0,5,5)):
        super(AttentiveEncoder, self).__init__()
        
        self.linf_max = linf_max
        self.data_dim = data_dim
        self.kernel_size = kernel_size
        self.padding = padding
  

        self._attention = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=kernel_size, padding=padding, stride=1),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, data_dim, kernel_size=kernel_size, padding=padding, stride=1),
            nn.Tanh(),
            nn.BatchNorm3d(data_dim),
        )
        self._conv = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=kernel_size, padding=padding, stride=1),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 3, kernel_size=kernel_size, padding=padding, stride=1),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        data = data * 2.0 - 1.0
        x = F.softmax(self._attention(frames), dim=1) # (N, D, L, H, W),softmax不改变维度
        x = torch.sum(multiplicative(x, data), dim=1, keepdim=True) # (N, 1, L, H, W)
        x = self._conv(torch.cat([frames, x], dim=1))
        return frames + self.linf_max * x

class AttentiveDecoder(nn.Module):
    """
    Input: (N, 3, L, H, W), (N, D,)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, encoder):
        super(AttentiveDecoder, self).__init__()
        self.data_dim = encoder.data_dim
        self._attention = encoder._attention
        self._conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=encoder.kernel_size, padding=encoder.padding, stride=1),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, self.data_dim, kernel_size=encoder.kernel_size, padding=encoder.padding, stride=1),
        )

    def forward(self, frames):
        N, D, L, H, W = frames.size()
        attention = F.softmax(self._attention(frames), dim=1) # (N, D, L, H, W)
        x = self._conv(frames) * attention
        return torch.mean(x.view(N, self.data_dim, -1), dim=2)

    if __name__ == '__main__':
        import time
        # device = 'cuda'
        encoder = AttentiveEncoder(96)#AttentiveEncoder(96).to(device)
        x = torch.rand([1,3,8,128,128])#.to(device)
        m = torch.randint(0, 2, (1,96))#.float().to(device)
        start = time.time()
        r = encoder(x, m)
        end = time.time()
        print (end - start)
        # d = decoder(y)
    