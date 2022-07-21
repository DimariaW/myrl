import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from myrl.model import Model
from models.layers import BertLayer, LayerNorm, MultiHeadAttentionLayer, activations

torch.set_num_threads(1)


class FootballEncoder(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.player_embedding = nn.Embedding(32, 5)  # padding_idx=0)
        self.mode_embedding = nn.Embedding(8, 3)   # padding_idx=0)
        self.fc_teammate = nn.Linear(23, filters)
        self.fc_opponent = nn.Linear(23, filters)
        self.fc = nn.Linear(filters + 41, filters)

    def forward(self, x):
        m_index = x['mode_index']  # 0-6
        # scalar features
        m_emb = self.mode_embedding(m_index)  # 3
        ball = x['ball']  # 9
        s = torch.cat([ball, x['match'], x['distance']['b2o'], m_emb], dim=-1)  # 9 + 16 + 6 + 3 = 34

        # player features
        p_emb_self = self.player_embedding(x['player_index']['self'])  # 11*5
        ball_concat_self = ball.unsqueeze(-2).expand(*p_emb_self.shape[:-1], -1)  # 11*9

        p_self = torch.cat([x['player']['self'], p_emb_self, ball_concat_self], dim=-1)  # 11*23

        p_emb_opp = self.player_embedding(x['player_index']['opp'])
        ball_concat_opp = ball.unsqueeze(-2).expand(*p_emb_opp.shape[:-1], -1)
        p_opp = torch.cat([x['player']['opp'], p_emb_opp, ball_concat_opp], dim=-1)

        # encoding linear layer
        p_self = self.fc_teammate(p_self)  # 11*96
        p_opp = self.fc_opponent(p_opp)  # 11*96

        p = F.relu(torch.cat([p_self, p_opp], dim=-2))  # 22*96
        s_concat = s.unsqueeze(-2).expand(*p.shape[:-1], -1)  # 22*34
        p = torch.cat([p, x['distance']['p2bo'], s_concat], dim=-1)  # 22*(96+34+7)

        h = F.relu(self.fc(p))

        return h


class FootballBlock(nn.Module):
    def __init__(self, filters, heads):
        super().__init__()
        self.bert_layer = BertLayer(filters, heads, dropout_rate=0.1, intermediate_size=(filters//heads) * 4,
                                    hidden_act='gelu', is_dropout=False)
        self.initializer_range = 0.5 / np.sqrt(filters)
        self.apply(self.init_model_weights)

    def forward(self, h):
        return self.bert_layer(h)

    def init_model_weights(self, module):
        """ 初始化权重
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # bert参数初始化, tf版本在linear和Embedding层使用的是截断正太分布, pytorch没有实现该函数,
            # 此种初始化对于加载预训练模型后进行finetune没有任何影响，
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class FootballControll(nn.Module):
    def __init__(self, filters, final_filters):
        super().__init__()
        self.filters = filters
        self.attention = MultiHeadAttentionLayer(hidden_size=filters, num_attention_heads=1, dropout_rate=0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.layer_norm1 = LayerNorm(filters)
        # self.fc_control = Dense(filters * 3, final_filters, bnunits=final_filters)
        self.fc_control = nn.Linear(filters * 2, final_filters)
        self.fc_gate = nn.Linear(filters * 2, final_filters)
        self.sigmoid = nn.Sigmoid()

        self.initializer_range = 0.5 / np.sqrt(filters)
        self.attention.apply(self.init_model_weights)
        self.layer_norm1.apply(self.init_model_weights)

    def forward(self, x, e, control_flag):
        x_controled = (x * control_flag).sum(dim=-2, keepdim=True)
        e_controled = (e * control_flag).sum(dim=-2, keepdim=True)

        h = self.attention(x_controled, x, x)
        h = self.layer_norm1(x_controled + self.dropout1(h))

        h = torch.cat([e_controled, h], dim=-1).squeeze(-2)
        # h = torch.cat([h, cnn_h.view(cnn_h.size(0), -1)], dim=1)
        h_control = self.fc_control(h)
        h_gate = self.sigmoid(self.fc_gate(h))

        return h_control * h_gate

    def init_model_weights(self, module):
        """ 初始化权重
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # bert参数初始化, tf版本在linear和Embedding层使用的是截断正太分布, pytorch没有实现该函数,
            # 此种初始化对于加载预训练模型后进行finetune没有任何影响，
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class FootballHead(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.head_p = nn.Linear(filters, 19, bias=False)
        self.head_p_special = nn.Linear(filters, 1 + 8 * 4, bias=False)
        self.head_v = nn.Linear(filters, 1, bias=True)
        self.head_r = nn.Linear(filters, 1, bias=False)

    def forward(self, x):
        p = self.head_p(x)
        #p2 = self.head_p_special(x)
        #v = self.head_v(x)
        r = self.head_r(x)
        return p, r#orch.cat([p, p2], -1), v, r


class CNNModel(nn.Module):
    def __init__(self, final_filters):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(53, 128, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 160, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 128, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.AdaptiveAvgPool2d((1, 11))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 160, kernel_size=(1, 1), stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(160),
            nn.Conv2d(160, 96, kernel_size=(1, 1), stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, final_filters, kernel_size=(1, 1), stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(final_filters),
        )
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, feature):
        shape = feature.shape
        feature = feature.view(-1, *shape[-3:])
        feature = self.conv1(feature)
        feature = self.pool1(feature)
        feature = self.conv2(feature)
        feature = self.pool2(feature)
        feature = self.flatten(feature)
        feature = feature.view(*shape[:-3], -1)
        return feature


class ActionHistoryEncoder(nn.Module):
    def __init__(self, input_size=19, hidden_size=64, num_layers=2, bidirectional=True):
        super().__init__()
        self.action_emd = nn.Embedding(19, 8)
        self.rnn = nn.GRU(8, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        h = self.action_emd(x)
        h = h.squeeze(dim=-2)
        self.rnn.flatten_parameters()
        shape = h.shape
        h = h.view(-1, *shape[-2:])
        h, _ = self.rnn(h)
        return h.view(*shape[:-1], -1)


class FootballNet(Model):
    def __init__(self):
        super().__init__()
        blocks = 2
        filters = 96
        final_filters = 128

        self.encoder = FootballEncoder(filters)
        self.blocks = nn.ModuleList([FootballBlock(filters, 8) for _ in range(blocks)])
        self.control = FootballControll(filters, final_filters)  # to head

        self.cnn = CNNModel(final_filters)  # to control
        rnn_hidden = 64
        self.rnn = ActionHistoryEncoder(19, rnn_hidden, 2)
        self.head = FootballHead(final_filters + final_filters + rnn_hidden * 2)
        # self.head = self.FootballHead(19, final_filters)

    def init_hidden(self, batch_size=None):
        return None

    def forward(self, x):
        feature = x["feature"]
        feature["mode_index"] = feature["mode_index"].squeeze(-1)
        e = self.encoder(feature)
        h = e
        for block in self.blocks:
            h = block(h)
        cnn_h = self.cnn(feature["cnn_feature"])
        #smm_h = self.smm(x)
        #h = self.control(h, e, x['control_flag'], cnn_h, smm_h)
        h = self.control(h, e, feature['control_flag'])
        rnn_h = self.rnn(feature['action_history'])

#         p, v, r = self.head(torch.cat([h,
#                                        cnn_h.view(cnn_h.size(0), -1),
#                                        smm_h.view(smm_h.size(0), -1)], axis=-1))

        rnn_h_head_tail = rnn_h[..., 0, :] + rnn_h[..., -1, :]
        rnn_h_plus_stick = torch.cat([rnn_h_head_tail[..., :-4], feature['control']], dim=-1)
        logit, value = self.head(torch.cat([h, cnn_h, rnn_h_plus_stick], dim=-1))

        legal_actions = x["legal_actions"]
        logit = logit - (1. - legal_actions) * 1e12

        return value.squeeze(-1), logit


class SimpleModel(Model):
    def __init__(self, num_left_players, num_right_players):
        super().__init__()

        self.ball_ebd = nn.Linear(9, 32)
        self.ball_owned_ebd = nn.Embedding(1 + num_left_players + num_right_players, 32)

        self.player_ebd = nn.Linear(4 * (num_left_players + num_right_players), 32)
        self.controlled_player_index_ebd = nn.Embedding(num_left_players, 32)

        self.game_mode_ebd = nn.Embedding(7, 32)

        self.ball_fc = nn.Linear(32, 64)
        self.player_fc = nn.Linear(32, 64)
        self.game_mode_fc = nn.Linear(32, 64)

        self.final_fc = nn.Linear(64, 20)

    def forward(self, obs):
        state = obs["state"]

        ball_embedding = self.ball_ebd(state["ball"])
        ball_owned_embedding = self.ball_owned_ebd(state["ball_owned"])

        player_embedding = self.player_ebd(state["player"])
        controlled_player_index_embedding = self.controlled_player_index_ebd(state["controlled_player_index"])

        ball_feas = ball_embedding + ball_owned_embedding + controlled_player_index_embedding
        ball_feas = ball_feas / np.sqrt(3)

        player_feas = player_embedding + controlled_player_index_embedding
        player_feas = player_feas / np.sqrt(2)

        ball_feature = self.ball_fc(ball_feas)
        player_feature = self.player_fc(player_feas)
        game_feature = self.game_mode_fc(self.game_mode_ebd(state["game_mode"]))

        feature = (ball_feature + player_feature + game_feature) / np.sqrt(3)

        out = self.final_fc(feature)

        value, logit = out[..., 0], out[..., 1:]

        legal_actions = obs["legal_actions"]
        logit = logit - (1. - legal_actions) * 1e12

        return value, logit











