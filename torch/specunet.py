import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import torchaudio.transforms as tat

def conv_block(nf_in, nf, ks):
    return nn.Sequential(        
        nn.Conv2d(nf_in, nf, ks, padding='same'),
        nn.ReLU(),
        nn.Conv2d(nf, nf, ks, padding='same'),
        nn.ReLU(),        
    )

def downconv_block(nf_in, nf, ks):
    return nn.Sequential(
        nn.Conv2d(nf_in, nf, ks, dilation=2, stride=2, padding=ks-1),
        nn.ReLU(),
        nn.Conv2d(nf, nf, ks, padding='same'),
        nn.ReLU(),
        nn.Conv2d(nf, nf, ks, padding='same'),
        nn.ReLU(),
    )

def upconv_block(nf_in, nf, ks):    
    return nn.Sequential(
        nn.ConvTranspose2d(nf_in, nf, ks, stride=2, padding=(ks-2), output_padding=1),
        nn.ReLU(),
        nn.Conv2d(nf, nf, ks, padding='same'),
        nn.ReLU(),
        nn.Conv2d(nf, nf, ks, padding='same'),
        nn.ReLU(),
    )

def out_block(nf_in, ks):
    return nn.Sequential(
        nn.Conv2d(nf_in, 2, ks, padding='same'),
        nn.ReLU(),
        nn.Conv2d(2, 2, 1, padding='same'),
        nn.Sigmoid(),
    )

class SpectrogramUNet(LightningModule):
    def __init__(self):
        super().__init__()

        self.ks = 3
        self.nfs = [ 32, 32, 64, 64, 128, 128, 256, 256 ]
        
        nlayers = len(self.nfs)

        self.b0 = conv_block(2, self.nfs[0], self.ks)

        self.down_blocks = nn.ModuleList([ downconv_block(self.nfs[i], self.nfs[i+1], ks=self.ks) for i in range(nlayers-1) ])
        self.up_blocks = nn.ModuleList([ upconv_block(2*self.nfs[-i-1], self.nfs[-i-2], ks=self.ks) for i in range(nlayers-1) ])

        self.out_block = out_block(self.nfs[0], ks=self.ks)

    def forward(self, x):        
        x = self.b0(x)

        outputs = [x]
        for d in self.down_blocks:
            x = d(x)
            outputs.append(x)

        for u,o in zip(self.up_blocks, outputs[::-1]):
            x = torch.cat((x,o),dim=1)
            x = nn.Dropout(0.2)(x)
            x = u(x)
        
        x = self.out_block(x)
                
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y, y_hat)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == "__main__":
    net = SpectrogramUNet()
    x = torch.randn(1, 2, 2**17)
    x = nn.Tanh()(x)
    print(x.shape)
    print(x.min(), x.max())
    out = net.forward(x)
    print(out.min(), out.max())

