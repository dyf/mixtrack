class MultiResBlock(nn.Module):
    def __init__(self, nf_in, nf, ks):
        super().__init__()

        outf = nf//6 + nf//3 + nf//2
        res = nf - outf
        
        self.b1 = nn.Sequential(
            nn.Conv1d(nf_in, nf//6, ks, padding='same'),
            nn.ReLU()
        )

        self.b2 = nn.Sequential(
            nn.Conv1d(nf//6, nf//3, ks, padding='same'),
            nn.ReLU()
        )

        self.b3 = nn.Sequential(
            nn.Conv1d(nf//3, nf//2+res, ks, padding='same'),
            nn.ReLU()
        )

        self.b0 = nn.Sequential(
            nn.Conv1d(nf_in, nf, 1, padding='same'),
            nn.ReLU()
        ) 

    def forward(self, x):
        x0 = self.b0(x)

        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)

        xout = torch.cat((x1,x2,x3), dim=1)
        
        return xout + x0    

class ResPathBlock(nn.Module):
    def __init__(self, layers, nf, ks):
        super().__init__()

        self.cs = nn.ModuleList([
                nn.Conv1d(nf, nf, ks, padding='same')
                for i in range(layers)
        ])
        self.rs = nn.ModuleList([
            nn.Conv1d(nf, nf, 1, padding='same')
            for i in range(layers)
        ])

    def forward(self, x):
        for c,r in zip(self.cs, self.rs):
            xr = r(x)
            x = c(x)
            x = xr + x
            
        return x

class MultiResUNet(LightningModule):
    def __init__(self):
        super().__init__()
        
        nfs =  [ 32, 48, 64, 72, 96, 128, 160 ]
        rbls = [  2,  2,  2,  2,  1,   1,   1 ]
        ks = 15
        layers = len(nfs)

        self.b0 = MultiResBlock(2, nfs[0], ks)

        self.dbs = nn.ModuleList([ MultiResBlock(nfs[i], nfs[i+1], ks) for i in range(layers-1) ])
        self.ds = nn.ModuleList([ nn.MaxPool1d(4) for i in range(layers-1) ])
        
        self.resbs = nn.ModuleList([ ResPathBlock(rbls[i], nfs[i], ks) for i in range(layers) ])
        self.drops = nn.ModuleList([ nn.Dropout(.2) for i in range(layers-1) ])

        self.ubs = nn.ModuleList([ MultiResBlock(nfs[-i-1]+nfs[-i-2], nfs[-i-2], ks) for i in range(layers-1) ])
        self.us = nn.ModuleList([ nn.Upsample(scale_factor=4) for i in range(layers-1) ])
        
        self.end = nn.Conv1d(nfs[0], 2, ks, padding='same')
        
    def forward(self, x):

        x = self.b0(x)
        
        outs = [self.resbs[0](x)]

        for db,d,rb,drop in zip(self.dbs, self.ds, self.resbs[1:], self.drops):
            x = d(x)            
            x = db(x)            
            outs.append(drop(rb(x)))
                    
        for ub,u,o in zip(self.ubs, self.us, outs[:-1][::-1]):
            x = u(x) 
            x = torch.cat((x, o), dim=1) # skip connections            
            x = ub(x)            
        
        x = self.end(x)

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
    net = MultiResUNet()    
    x = torch.randn(1, 2, 4096)
    print(x.shape)
    out = net.forward(x)
    print(out.shape)
    print(net)

    
