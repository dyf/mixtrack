import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

def downconv_block(nf_in, nf, ks):
    return nn.Sequential(        
        nn.Conv1d(nf_in, nf, ks, dilation=4, stride=4, padding=2*(ks-1)),        
        nn.ReLU(),
        nn.Conv1d(nf, nf, ks, padding='same'),
        nn.ReLU(),
        nn.Conv1d(nf, nf, ks, padding='same')
    )

def upconv_block(nf_in, nf, ks):    
    return nn.Sequential(
        nn.ConvTranspose1d(nf_in, nf, ks, stride=4, padding=(ks-1)//2-1, output_padding=1),
        nn.ReLU(),
        nn.Conv1d(nf, nf, ks, padding='same'),
        nn.ReLU(),
        nn.Conv1d(nf, nf, ks, padding='same'),
    )

def out_block(ks):
    return nn.Sequential(
        nn.Conv1d(2, 2, ks, padding='same'),
        nn.ReLU(),
        nn.Conv1d(2, 2, 1)
    )

class UNet(LightningModule):
    def __init__(self):
        super().__init__()

        self.ks = 13
        self.nfs = [ 2, 32, 32, 32, 32, 32, 32, 32, 32 ]
        
        nlayers = len(self.nfs)

        self.down_blocks = nn.ModuleList([ downconv_block(self.nfs[i], self.nfs[i+1], ks=self.ks) for i in range(nlayers-1) ])
        self.up_blocks = nn.ModuleList([ upconv_block(2*self.nfs[-i-1], self.nfs[-i-2], ks=self.ks) for i in range(nlayers-1) ])
        self.out_block = out_block(ks=self.ks)

    def forward(self, x):
        outputs = []
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

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

