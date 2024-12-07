
import torch
import torch.nn as nn

class GraphRegConvNet(nn.Module):
    def __init__(self, input_size = 100_000, dp_rate=0.5, return_hidden=False, **kwargs): #take in kwargs for other useless things, we don't use them tho
        super(GraphRegConvNet, self).__init__()
        
        self.input_size = input_size
        self.dp_rate = dp_rate
        self.return_hidden = return_hidden
        
        self.conv1 = nn.Conv1d(4,256,21, bias=True, padding='same')
        self.bn1 = nn.BatchNorm1d(256)
        self.mp1 = nn.MaxPool1d(2)
        self.dp1 = nn.Dropout(dp_rate)
        
        self.conv2 = nn.Conv1d(256,128,3, bias=True, padding='same')
        self.bn2 = nn.BatchNorm1d(128)
        self.mp2 = nn.MaxPool1d(2)
        self.dp2 = nn.Dropout(dp_rate)

        self.conv3 = nn.Conv1d(128,128,3, bias=True, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        self.mp3 = nn.MaxPool1d(5)
        self.dp3 = nn.Dropout(dp_rate)
        
        self.conv4 = nn.Conv1d(128,128,3, bias=True, padding='same')
        self.bn4 = nn.BatchNorm1d(128)
        self.mp4 = nn.MaxPool1d(5)
        self.dp4 = nn.Dropout(dp_rate)
        
        self.conv5 = nn.Conv1d(128,64,3, bias=True, padding='same')
        self.bn5 = nn.BatchNorm1d(64)
        self.dp5 = nn.Dropout(dp_rate)
        
        #dilated convs
        for i in range(1,7):
            self.add_module(f'conv6_{i}', nn.Conv1d(64,64,3, dilation=2**i, padding='same'))
            self.add_module(f'bn6_{i}', nn.BatchNorm1d(64))
            self.add_module(f'dp6_{i}', nn.Dropout(dp_rate))
        self.dp6_6 = nn.Dropout(0) #there's no final dropout, so remove it
            
        self.conv_me3 = nn.Conv1d(64, 1, 5, padding='same')
        self.conv_27ac = nn.Conv1d(64, 1, 5, padding='same')
        self.conv_dnase = nn.Conv1d(64, 1, 5, padding='same')
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 4, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, input_size//100, 3)
        """
        # print('training:', self.training)
        # print('input shape:', x.shape) #these shapes are all what you would expect
        x = self.conv1(x).relu()
        # print('conv1 shape:', x.shape)
        x = self.bn1(x)
        # print('bn1 shape:', x.shape)
        # print('max:', x.max()) generally like 24 or so, not an issue with the 
        # if not self.training:
        #     torch.save(x.cpu(), '/data1/lesliec/sarthak/data/GraphReg/temp_compare/x.pt')
        x = self.mp1(x)
        # print('mp1 shape:', x.shape)
        x = self.dp1(x)
        
        x = self.conv2(x).relu()
        x = self.bn2(x)
        x = self.mp2(x)
        x = self.dp2(x)
        
        x = self.conv3(x).relu()
        x = self.bn3(x)
        x = self.mp3(x)
        x = self.dp3(x)
        
        x = self.conv4(x).relu()
        x = self.bn4(x)
        x = self.mp4(x)
        x = self.dp4(x)
        
        x = self.conv5(x).relu()
        h = self.bn5(x)
        x = h
        x = self.dp5(x)
        
        for i in range(1,7):
            x = self._modules[f'conv6_{i}'](x).relu() + x
            x = self._modules[f'bn6_{i}'](x)
            x = self._modules[f'dp6_{i}'](x)
            
        out_me3 = self.conv_me3(x).exp().view(-1,self.input_size//100)
        out_27ac = self.conv_27ac(x).exp().view(-1,self.input_size//100)
        out_dnase = self.conv_dnase(x).exp().view(-1,self.input_size//100)
        
        #stack them together in a tensor along a new third dimension
        out = torch.stack((out_me3, out_27ac, out_dnase), dim=2)

        if self.return_hidden:
            return out, h
        else:
            return out