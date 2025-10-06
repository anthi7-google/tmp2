import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    No Decomposition-CNN
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in

        self.cnn_out_channel = 16
        self.cnn_kernel_size = 8
        self.CNN_Main = nn.Conv1d(configs.enc_in-1, self.cnn_out_channel, self.cnn_kernel_size, padding='same')
            
        self.CNN_Main_2 =nn.Conv1d(self.cnn_out_channel, 2*self.cnn_out_channel, self.cnn_kernel_size, padding='same')

        self.cnn_out_2_size = self.seq_len

        self.Linear_Main = nn.Linear(2*self.cnn_out_channel, 1)

    def forward(self, x):
        #print("x:size = ", x.shape)
        # x: [Batch, Input length, Channel]
        main_init = x
        main_init = main_init.permute(0,2,1)
        # main_init: [Batch, Channel, Input length]

        main_cnn = F.relu(self.CNN_Main(main_init))
        main_cnn = F.relu(self.CNN_Main_2(main_cnn))
        
        main_cnn = main_cnn.permute(0,2,1)

        output = self.Linear_Main(main_cnn)

        x = output

        return x
