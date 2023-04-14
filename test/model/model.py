import torch
from torch import nn



class Try(nn.Module):
    def __init__(self, seq, batch_size, scale=0):
        super(Try, self).__init__()
        self.scale = scale
        self.seq = seq
        self.batch_size = batch_size
        self.linear = nn.Sequential(
            nn.Linear((self.seq+1)*12, (self.seq+1)*6),
            nn.Dropout(0.5),
            nn.Sigmoid(),

            nn.Linear((self.seq+1)*6, (self.seq+1)*6),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),

        )
        self.conv1 = nn.Sequential(
            # seq * 5 
            nn.Conv2d(1, 2, kernel_size=(3,3), padding=2, bias=False), # (seq+2) * 7
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(2, 2, kernel_size=(3,3), padding=1, bias=False), # (seq+2) * 7
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=1), # (seq+1) * 6
            
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(2,2), padding=0, bias=False), # seq * 5
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(2, 1, kernel_size=(1,1), padding=0, bias=True), # seq * 5
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        out = self.conv1(x)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)
        # print(out.shape)
        with torch.no_grad():
            out = out.reshape(self.batch_size, 1, self.seq+1, 6)
        out = self.conv1(out)
        assert out.shape==x.shape, "Shape Unequal Error."
        return out + x
    


class NET(nn.Module):
    def __init__(self, seq, batch_size, ablation_scale=1.0):
        super(NET, self).__init__()
        self.seq = seq
        self.batch_size = batch_size
        self.scale = ablation_scale
        self.input_size = self.hidden_size = seq
        
        self.is_directional = False
        self.num_direction = 2 if self.is_directional else 1
        self.num_layers = 1
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=0.5, bidirectional=self.is_directional, batch_first=False)
        
        self.decouple = nn.Sequential(
            nn.Linear(self.seq*5,self.seq*5),  # linear decouple
            nn.Dropout(0.5),
#             nn.Sigmoid(),   # use sigmoid to enhance unlinear regression may cause data shifting
            nn.Linear(self.seq*5, self.seq*5),     # unlinear pool
        )
        self.conv = nn.Sequential(
            # seq * 5 
            nn.Conv2d(1, 2, kernel_size=(3,3), padding=1, bias=False), # seq * 5
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(2, 2, kernel_size=(1,1), padding=0, bias=False),  # seq * 5
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(2, 1, kernel_size=(3,3), padding=1, bias=True), # seq * 5
            nn.ReLU(inplace=True),
        )
    
    def reset(self, scale=1.0):
        self.ablatiion_scale = scale

    def forward(self, x):
        x = x.squeeze()
        
        h_0 = torch.randn(self.num_direction*self.num_layers, self.seq, self.hidden_size).cuda()
        c_0 = torch.randn(self.num_direction*self.num_layers, self.seq, self.hidden_size).cuda()
        pred, *_ = self.lstm(x, (h_0, c_0))
        pred = pred.flatten(1)
        out1 = self.decouple(pred)
        with torch.no_grad():
            out1 = out1.reshape(-1, 1, self.seq, 5)
        out2 = self.conv(out1)
        assert out1.shape==out2.shape, "Shape Unequal Error."
        return self.scale * out1 + out2
