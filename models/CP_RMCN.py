import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm


def build_main_idx_by_count(base=6, count=3):
    fixed = {0, 1, 2, 3}
    dynamic = {base * (2 ** i) for i in range(count)}
    return fixed | dynamic


class RSDM(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, resolution, seq_len):
        super().__init__()
        self.seq_len = seq_len
        if resolution == 'hour':#336h=2weeks
            self.main_idx = {0, 1, 2, 7, 14, 28, 56} #2w, 1w, 2d, 1d, 12h, 6h
        elif resolution == 'min':#15mins
            self.main_idx = {0, 1, 2, 3, 7, 14, 28} #3.5h,
        elif resolution == '10min':
            self.main_idx = {0, 1, 2, 5, 9, 19, 56} #56h, 1d, 12h, 6h, 3h, 1h
        elif resolution == 'week': #48weeks
            self.main_idx = {0, 1, 2, 3, 4, 6, 8}
            #build_main_idx_by_count(base=4)

    def forward(self, x):
        B, L, C = x.shape
        xf = torch.fft.rfft(x, dim=1)
        F = xf.shape[1]

        mask = torch.zeros_like(xf)

        keep_idx = [i for i in self.main_idx if i < F]
        idx_tensor = torch.tensor(keep_idx, device=x.device).view(1, -1, 1)  #(1, K, 1)
        idx_tensor = idx_tensor.expand(B, -1, C)  #(B, K, C)

        mask.scatter_(1, idx_tensor, 1)

        xf_prd = xf * mask

        x_prd = torch.fft.irfft(xf_prd, n=L, dim=1).real.float()

        res = x - x_prd
        return res, x_prd


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition 
        #kernel_size = 25
        self.decompsition = RSDM(configs.resolution, configs.seq_len)
        self.channels = configs.enc_in

        self.cnn_out_channel = 16
        
        self.DBPN_Res = DBPN(configs, out_ch = self.cnn_out_channel)
        self.DBPN_Prd = DBPN(configs, out_ch = self.cnn_out_channel)
     
        self.Linear_Res = nn.Linear(2*self.cnn_out_channel, 1)
        self.Linear_Prd = nn.Linear(2*self.cnn_out_channel, 1)

    def forward(self, x, tr = False):
        #print("x:size = ", x.shape)
        # x: [Batch, Input length, Channel]
        res_init, prd_init = self.decompsition(x)
        res_init, prd_init = res_init.permute(0,2,1), prd_init.permute(0,2,1)
        # res_init: [Batch, Channel, Input length]
        # prd_init: [Batch, Channel, Input length]

        res_cnn = self.DBPN_Res(res_init)
        prd_cnn = self.DBPN_Prd(prd_init)
        
        res_cnn, prd_cnn = res_cnn.permute(0,2,1), prd_cnn.permute(0,2,1)

        res_output = self.Linear_Res(res_cnn)
        prd_output = self.Linear_Prd(prd_cnn)

        x = res_output + prd_output

        if tr: return x, res_output, prd_output
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class DilatedBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class DilatedConv(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [DilatedBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    
class DBPN(nn.Module):
    def __init__(self, configs, out_ch = 16):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.channels = configs.enc_in
        tcn_layers = [4,4,0]

        self.cnn_out_channel = out_ch
        self.cnn_kernel_size_coarse = 16
        self.cnn_kernel_size_fine = 8

        self.cnn_kernel_size_emb = 8          
        self.in_embed = nn.Conv1d(configs.enc_in-1, self.cnn_out_channel, self.cnn_kernel_size_emb, padding='same')
        
        self.DiCNN = DilatedConv(self.cnn_out_channel, [self.cnn_out_channel]*tcn_layers[0]+[2*self.cnn_out_channel]*tcn_layers[1]+[4*self.cnn_out_channel]*tcn_layers[2])
        
        ch_l0 = self.cnn_out_channel
        ch_l1 = self.cnn_out_channel
        ch_l2 = 2*self.cnn_out_channel
        ch_l3 = 2*self.cnn_out_channel
        ch_l4 = 2*self.cnn_out_channel

        self.Conv_1_coarse = nn.Conv1d(ch_l0, ch_l1, self.cnn_kernel_size_coarse, padding='same')
        self.Conv_2_coarse = nn.Conv1d(ch_l1, ch_l2, self.cnn_kernel_size_coarse, padding='same')
        self.Conv_3_coarse = nn.Conv1d(ch_l2, ch_l3, self.cnn_kernel_size_coarse, padding='same')
        self.Conv_4_coarse = nn.Conv1d(ch_l3, ch_l4, self.cnn_kernel_size_coarse, padding='same')

        self.Conv_1_fine = nn.Conv1d(ch_l0, ch_l1, self.cnn_kernel_size_fine, padding='same')
        self.Conv_2_fine = nn.Conv1d(ch_l1, ch_l2, self.cnn_kernel_size_fine, padding='same')
        self.Conv_3_fine = nn.Conv1d(ch_l2, ch_l3, self.cnn_kernel_size_fine, padding='same')
        self.Conv_4_fine = nn.Conv1d(ch_l3, ch_l4, self.cnn_kernel_size_fine, padding='same')        

        self.CIU_1 = CIU(ch_l1)
        self.CIU_2 = CIU(ch_l2)
        self.CIU_3 = CIU(ch_l3)
        self.CIU_4 = CIU(ch_l4)
        
    def forward(self, x):
        #print("x:size = ", x.shape)
        # x: [Batch, Channel, Input length]
        x_emb = self.in_embed(x)
        
        x_dcnn = self.DiCNN(x_emb)       
        
        x_cnn_c = self.Conv_1_coarse(x_emb)
        x_cnn_f = self.Conv_1_fine(x_emb)
        x_cnn_c, x_cnn_f = self.CIU_1(x_cnn_c, x_cnn_f)
        x_cnn_c, x_cnn_f = F.relu(x_cnn_c), F.relu(x_cnn_f)

        x_cnn_c = self.Conv_2_coarse(x_cnn_c)
        x_cnn_f = self.Conv_2_fine(x_cnn_f)
        x_cnn_c, x_cnn_f = self.CIU_2(x_cnn_c, x_cnn_f)
        x_cnn_c, x_cnn_f = F.relu(x_cnn_c), F.relu(x_cnn_f)

        x_cnn_c = self.Conv_3_coarse(x_cnn_c)
        x_cnn_f = self.Conv_3_fine(x_cnn_f)
        x_cnn_c, x_cnn_f = self.CIU_3(x_cnn_c, x_cnn_f)
        x_cnn_c, x_cnn_f = F.relu(x_cnn_c), F.relu(x_cnn_f)

        x_cnn_c = self.Conv_4_coarse(x_cnn_c)
        x_cnn_f = self.Conv_4_fine(x_cnn_f)
        x_cnn_c, x_cnn_f = self.CIU_4(x_cnn_c, x_cnn_f)
        x_cnn_c, x_cnn_f = F.relu(x_cnn_c), F.relu(x_cnn_f)

        x_cnn = x_cnn_c + x_cnn_f

        return x_dcnn + x_cnn 
    
class CIU(nn.Module):
    def __init__(self, out_ch = 16):
        super().__init__()
        self.Linear = nn.Linear(2*out_ch, 2*out_ch, bias=False)
        self.split_sizes = [out_ch, out_ch]
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        x_cat = x1.permute(0,2,1), x2.permute(0,2,1)
        x_cat = torch.cat(x_cat, dim=-1)
        w = F.softmax(self.alpha)
        x_cat = w*x_cat + (1-w)*self.Linear(x_cat)
        x1, x2 = torch.split(x_cat, split_size_or_sections=self.split_sizes, dim=2)
        x1, x2 = x1.permute(0,2,1), x2.permute(0,2,1)
        return x1, x2