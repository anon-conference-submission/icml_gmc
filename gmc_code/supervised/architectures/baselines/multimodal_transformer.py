import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from gmc_code.supervised.architectures.baselines.transformer_networks import TransformerEncoder


class MULTModel(LightningModule):
    def __init__(self, scenario):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.name = 'multimodal_transformer'
        if scenario == 'mosei':
            self.orig_d_l, self.orig_d_a, self.orig_d_v = 300, 74, 35
            self.d_l, self.d_a, self.d_v = 30, 30, 30
            self.vonly = True
            self.aonly = True
            self.lonly = True
            self.num_heads = 5
            self.layers = 5
            self.attn_dropout = 0.1
            self.attn_dropout_a = 0.0
            self.attn_dropout_v = 0.0
            self.relu_dropout = 0.1
            self.res_dropout = 0.1
            self.out_dropout = 0.0
            self.embed_dropout = 0.25
            self.attn_mask = True
        else:
            self.orig_d_l, self.orig_d_a, self.orig_d_v = 300, 5, 20
            self.d_l, self.d_a, self.d_v = 30, 30, 30
            self.vonly = True
            self.aonly = True
            self.lonly = True
            self.num_heads = 10
            self.layers = 5
            self.attn_dropout = 0.2
            self.attn_dropout_a = 0.0
            self.attn_dropout_v = 0.0
            self.relu_dropout = 0.1
            self.res_dropout = 0.1
            self.out_dropout = 0.0
            self.embed_dropout = 0.25
            self.attn_mask = True

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l  # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        output_dim = 1

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        # Training
        self.scenario = scenario
        self.criterion = nn.L1Loss()

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x):

        x_l, x_a, x_v = x[0].cuda(), x[1].cuda(), x[2].cuda()

        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output, last_hs

    def encode(self, x, sample=False):

        # Get batch-size
        for i in range(len(x)):
            if x[i] is not None:
                batch_size = x[i].size(0)
                break

        x_l, x_a, x_v = x[0].cuda(), x[1].cuda(), x[2].cuda()

        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """

        # if x_l is None:
        #     x_l = torch.zeros([batch_size, 50, 300])
        #
        # if x_a is None:
        #     x_a = torch.zeros([batch_size, 50, 74])
        #
        # if x_v is None:
        #     x_v = torch.zeros([batch_size, 50, 35])

        if x_l is not None:
            x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
            proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
            proj_x_l = proj_x_l.permute(2, 0, 1)
        else:
            proj_x_l = None

        if x_a is not None:
            x_a = x_a.transpose(1, 2)
            proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
            proj_x_a = proj_x_a.permute(2, 0, 1)
        else:
            proj_x_a = None

        if x_v is not None:
            x_v = x_v.transpose(1, 2)
            proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
            proj_x_v = proj_x_v.permute(2, 0, 1)
        else:
            proj_x_v = None


        if self.lonly and proj_x_l is not None:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        else:
            last_h_l = torch.zeros([batch_size, 60]).cuda()

        if self.aonly and proj_x_a is not None:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]
        else:
            last_h_a = torch.zeros([batch_size, 60]).cuda()


        if self.vonly and proj_x_v is not None:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        else:
            last_h_v = torch.zeros([batch_size, 60]).cuda()

        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1).cuda()

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output



    def training_step(self, data, target_data, train_params):

        # Forward pass through the encoders
        preds, _ = self.forward(data)

        if self.scenario == 'iemocap':
            preds = preds.view(-1, 2)
            target_data = target_data.view(-1)

        loss = self.criterion(preds, target_data)

        tqdm_dict = {'loss': loss}

        return loss, tqdm_dict

    def validation_step(self, data, target_data, train_params):

        # Forward pass through the encoders
        preds, _ = self.forward(data)

        if self.scenario == 'iemocap':
            preds = preds.view(-1, 2)
            target_data = target_data.view(-1)

        loss = self.criterion(preds, target_data)

        tqdm_dict = {'loss': loss}

        return tqdm_dict