import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

###############################################################################

def tokenize(smiles_list):
    
    max_length = np.max([len(x) for x in smiles_list])
    pad_char = "_"
    
    for i in range(len(smiles_list)):
        if len(smiles_list[i]) < max_length:
            smiles_list[i] += pad_char * (max_length - len(smiles_list[i]))
    
    token_to_id = {}
    next_id = 0
    for smile in smiles_list:
        for token in smile:
            if token not in token_to_id:
                token_to_id[token] = next_id
                next_id += 1
    
    x = np.zeros((len(smiles_list), max_length, len(token_to_id)))
    
    for i in range(len(smiles_list)):
        for j in range(max_length):
            token_id= token_to_id[smiles_list[i][j]]
            x[i,j,token_id] = 1
        
    return x 

#-----------------------------------------------------------------------------#

class VAEDataset(Dataset):

    def __init__(self,
                 x
                 ):
        
        x = np.swapaxes(x, 2, 1)
        
        self.x_i = x[:, :, :-1]
        self.x_o = x[:, :, 1:]

    def __len__(self):
        """
        Method necessary for Pytorch training
        """
        return len(self.x_i)

    def __getitem__(self, idx):
        """
        Method necessary for Pytorch training
        """
        
        x_i = torch.tensor(self.x_i[idx], dtype=torch.float32)
        x_o = torch.tensor(self.x_o[idx], dtype=torch.float32)

        return x_i, x_o
    
#-----------------------------------------------------------------------------#

class VAE:
        
    def __init__(self,
                 dict_size,
                 max_length,
                 num_epochs = 40,
                 num_delay = 10,
                 device = "auto",
                 ):
        
        self.dict_size = dict_size
        self.max_length = max_length
        self.kl_weight = 0
        self.num_epochs = num_epochs
        self.num_delay = num_delay
        self.num_anneal = num_epochs - num_delay
        self.anneal_increase = torch.tensor(1 / self.num_anneal, dtype=torch.float32)
        self.kl_sum = 0
                
        d1 = max_length - 9 + 1
        d2 = d1 - 9 + 1
        d3 = d2 - 11 + 1
        
        self.encoder = nn.Sequential(*[
                nn.Conv1d(in_channels=dict_size, out_channels=9, kernel_size=9),
                nn.BatchNorm1d(9),
                nn.Tanh(),
                nn.Conv1d(in_channels=9, out_channels=9, kernel_size=9),
                nn.BatchNorm1d(9),
                nn.Tanh(),
                nn.Conv1d(in_channels=9, out_channels=10, kernel_size=11),
                nn.BatchNorm1d(10),
                nn.Tanh(),
                nn.Flatten(),
                ])

        self.z_mean = nn.Linear(in_features=d3*10, out_features=196)
        self.z_std = nn.Linear(in_features=d3*10, out_features=196)
        
        self.decoder = nn.GRU(input_size=dict_size+196, hidden_size=488, 
                              num_layers=3, batch_first=True, 
                              dropout=0.19617749608323892)
        
        self.final = nn.Linear(in_features=488, out_features=dict_size)
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.z_mean.parameters()) +
            list(self.z_std.parameters()) +
            list(self.decoder.parameters()) +
            list(self.final.parameters()),
            lr=0.00039192162392520126,
            betas=(0.97170900638688007, 0.999)
            )
        
        if device == "auto":
        	self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
        	self.device = device
        self.move_to_device()
        
        self.N = torch.distributions.Normal(0, 1)
        if self.device != "cpu":
        	self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        	self.N.scale = self.N.scale.cuda()
        
    def forward(self, x_i, is_training, deterministic):
        
        # switch according to input
        self.switch_mode(is_training)
        
        encoded = self.encoder(x_i)
        mean = self.z_mean(encoded)
        std = torch.exp(self.z_std(encoded) / 2)
        if deterministic is False:
            z = mean + std*self.N.sample(mean.shape)
        else:
            z = mean
        kl =  (std**2 + mean**2 - torch.log(std) - 1).sum()
        kl = self.kl_weight * kl
        
        z = z.unsqueeze(2)
        z = z.repeat(1,1,self.max_length)
        z_cat = torch.cat((x_i, z), axis=1)
        z_cat = z_cat.permute(0,2,1)
        
        decoded = self.decoder(z_cat)[0]
        
        out = self.final(decoded)
        out = out.permute(0,2,1)
        
        return out, kl
    
    def anneal_kl_weight(self, epoch):
        if epoch < self.num_delay:
            self.kl_weight = 0
        else:
            current_epoch = epoch - self.num_delay
            exponent = ((self.num_anneal / 2) - current_epoch)
            self.kl_weight = 1 / (1 + math.exp(exponent))
    
    def train(self,
                  train_data: torch.utils.data.DataLoader):

        cross_entropy = torch.nn.CrossEntropyLoss()
        
        for epoch in range(self.num_epochs):
            # initialize loss containers
            rec_loss = 0.0
            kl_loss = 0.0
            
            # loop over training set
            for batch_idx, (x_i, x_o) in enumerate(train_data):
                
                # reset grad
                self.optimizer.zero_grad()

                # send samples to device
                x_i = x_i.to(self.device)
                x_o = x_o.to(self.device)

                # compute train loss for batch
                x_rec, kl = self.forward(x_i, is_training=True,
                			deterministic=False)
                rec = cross_entropy(x_rec, x_o)
                total_loss = rec + kl

                # backpropagation and optimization
                total_loss.backward()
                self.optimizer.step()
                
                # add i-th loss to training container
                rec_loss += rec.item()
                kl_loss += kl.item()
            
            # increase kl weight
            self.anneal_kl_weight(epoch)
            
    def predict(self,
                  x: np.ndarray):
        
        x = np.swapaxes(x, 2, 1)
        
        x_i = torch.tensor(x[:, :, :-1], dtype=torch.float32)
        x_o = torch.tensor(x[:, :, 1:], dtype=torch.float32)
        
        cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
               
        # send samples to device
        x_i = x_i.to(self.device)
        x_o = x_o.to(self.device)

        # compute reconstruction error
        x_rec, kl = self.forward(x_i, is_training=False, deterministic=True)
        rec = cross_entropy(x_rec, x_o).cpu().detach().numpy()
        
        return np.mean(rec, axis=1)

    def move_to_device(self):
        # move each Sequential model to device
        self.encoder.to(self.device)
        self.z_mean.to(self.device)
        self.z_std.to(self.device)
        self.decoder.to(self.device)
        self.final.to(self.device)

    def switch_mode(self, is_training: bool):
        if is_training is True:
            self.encoder.train()
            self.z_mean.train()
            self.z_std.train()
            self.decoder.train()
            self.final.train()
        else:
            self.encoder.eval()
            self.z_mean.eval()
            self.z_std.eval()
            self.decoder.eval()
            self.final.eval()




