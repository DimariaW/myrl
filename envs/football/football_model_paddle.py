import paddle
import parl
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import gfootball.env as gfootball_env


def to_tensor(x, unsqueeze=None, place=paddle.device.set_device("cpu")):
    if isinstance(x, (list, tuple, set)):
        return type(x)(to_tensor(xx, unsqueeze) for xx in x)
    elif isinstance(x, dict):
        return type(x)((key, to_tensor(xx, unsqueeze)) for key, xx in x.items())
    elif isinstance(x, np.ndarray):
        if x.dtype == np.int32 or x.dtype == np.int64:
            t = paddle.to_tensor(x, dtype="int64", place=place)
        else:
            t = paddle.to_tensor(x, dtype="float32", place=place)
        return t if unsqueeze is None else t.unsqueeze(unsqueeze)


class FootballNet(parl.Model):
    def __init__(self):
        super().__init__()
        blocks = 5
        filters = 96
        final_filters = 128

        self.encoder = FootballEncoder(filters)
        self.blocks = nn.LayerList([FootballBlock(filters, 8) for _ in range(blocks)])
        self.control = FootballControl(filters, final_filters)  # to head

        self.cnn = CNNModel(final_filters)  # to control
        rnn_hidden = 64
        self.rnn = ActionHistoryEncoder(19, rnn_hidden, 2)
        self.head = FootballHead(final_filters + final_filters + rnn_hidden * 2)
        # self.head = self.FootballHead(19, final_filters)

    def forward(self, x):
        state = x['state']
        e = self.encoder(state)
        h = e
        for block in self.blocks:
            h = block(h)
        h = self.control(h, e, state['control_flag'])

        cnn_h = self.cnn(state)
        rnn_h = self.rnn(state)

        rnn_h_head_tail = rnn_h[:, 0, :] + rnn_h[:, -1, :]
        rnn_h_plus_stick = paddle.concat([rnn_h_head_tail[:, :-4], state['control']], axis=1)

        fea = paddle.concat([h, cnn_h, rnn_h_plus_stick], axis=-1)
        logits, r = self.head(fea)
        legal_actions = x["legal_actions"]
        logits = logits - (1. - legal_actions) * 1e12

        return logits, r

#%%


class FootballEncoder(nn.Layer):
    def __init__(self, filters):
        super().__init__()
        self.player_embedding = nn.Embedding(32, 5, padding_idx=0)
        self.mode_embedding = nn.Embedding(8, 3, padding_idx=0)
        self.fc_teammate = nn.Linear(23, filters)
        self.fc_opponent = nn.Linear(23, filters)
        self.fc = nn.Linear(filters + 41, filters)

    def forward(self, x):
        bs = x['mode_index'].shape[0]
        # scalar features
        m_emb = self.mode_embedding(x['mode_index']).reshape((bs, -1))
        ball = x['ball']
        s = paddle.concat([ball, x['match'], x['distance']['b2o'], m_emb], axis=1)

        # player features
        p_emb_self = self.player_embedding(x['player_index']['self'])
        ball_concat_self = ball.reshape((bs, 1, -1)).tile(repeat_times=[1, 11, 1])
        p_self = paddle.concat([x['player']['self'], p_emb_self, ball_concat_self], axis=2)

        p_emb_opp = self.player_embedding(x['player_index']['opp'])
        ball_concat_opp = ball.reshape((bs, 1, -1)).tile(repeat_times=[1, 11, 1])
        p_opp = paddle.concat([x['player']['opp'], p_emb_opp, ball_concat_opp], axis=2)

        # encoding linear layer
        p_self = self.fc_teammate(p_self)
        p_opp = self.fc_opponent(p_opp)

        p = F.relu(paddle.concat([p_self, p_opp], axis=1))
        s_concat = s.reshape((bs, 1, -1)).tile(repeat_times=[1, 22, 1])
        p = paddle.concat([p, x['distance']['p2bo'], s_concat], axis=2)

        h = F.relu(self.fc(p))

        return h


class MultiHeadAttention(nn.Layer):
    def __init__(self, in_dim, out_dim, out_heads, relation_dim=0,
                 residual=True, projection=True, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_heads = out_heads
        self.relation_dim = relation_dim
        assert self.out_dim % self.out_heads == 0

        weight_attr_q = paddle.ParamAttr(initializer=nn.initializer.Uniform(-0.1, 0.1))
        self.query_layer = nn.Linear(self.in_dim + self.relation_dim, self.out_dim,
                                     weight_attr=weight_attr_q, bias_attr=False)

        weight_attr_k = paddle.ParamAttr(initializer=nn.initializer.Uniform(-0.1, 0.1))
        self.key_layer = nn.Linear(self.in_dim + self.relation_dim, self.out_dim,
                                   weight_attr=weight_attr_k,  bias_attr=False)

        weight_attr_v = paddle.ParamAttr(initializer=nn.initializer.Uniform(-0.1, 0.1))
        self.value_layer = nn.Linear(self.in_dim, self.out_dim,
                                     weight_attr=weight_attr_v, bias_attr=False)

        self.residual = residual
        self.projection = projection
        if self.projection:
            weight_attr_p = paddle.ParamAttr(initializer=nn.initializer.Uniform(-0.1, 0.1))
            self.proj_layer = nn.Linear(self.out_dim, self.out_dim, weight_attr=weight_attr_p)

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln = nn.LayerNorm(self.out_dim)

    def forward(self, query, key, relation=None, mask=None, key_mask=None, distance=None):
        """
        Args:
            query (torch.Tensor): [batch, query_len, in_dim]
            key (torch.Tensor): [batch, key_len, in_dim]
            relation (torch.Tensor): [batch, query_len, key_len, relation_dim]
            mask (torch.Tensor): [batch, query_len]
            key_mask (torch.Tensor): [batch, key_len]
        Returns:
            torch.Tensor: [batch, query_len, out_dim]
        """
        query_len = query.shape[-2]
        key_len = key.shape[-2]
        head_dim = self.out_dim // self.out_heads

        Q = self.query_layer(query).reshape((-1, query_len, self.out_heads, head_dim))
        K = self.key_layer(key).reshape((-1, key_len, self.out_heads, head_dim))

        Q = Q.transpose(perm=[0, 2, 1, 3]).reshape((-1, query_len, head_dim))
        K = K.transpose(perm=[0, 2, 1, 3]).reshape((-1, key_len, head_dim))

        attention = paddle.bmm(Q, K.transpose(perm=[0, 2, 1]))

        attention = attention * (float(head_dim) ** -0.5)

        attention = F.softmax(attention, axis=-1)

        V = self.value_layer(key).reshape((-1, key_len, self.out_heads, head_dim))
        V = V.transpose(perm=[0, 2, 1, 3]).reshape((-1, key_len, head_dim))

        output = paddle.bmm(attention, V).reshape((-1, self.out_heads, query_len, head_dim))
        output = output.transpose(perm=[0, 2, 1, 3]).reshape((-1, query_len, self.out_dim))

        if self.projection:
            output = self.proj_layer(output)

        if self.residual:
            output = output + query

        if self.layer_norm:
            output = self.ln(output)

        return output


class FootballBlock(nn.Layer):
    def __init__(self, filters, heads):
        super().__init__()
        self.attention = MultiHeadAttention(filters, filters, heads, relation_dim=0,
                                            residual=True, projection=True)

    def forward(self, x, rel=None, distance=None):
        h = self.attention(x, x, relation=None, distance=None)
        return h


class FootballControl(nn.Layer):
    def __init__(self, filters, final_filters):
        super().__init__()
        self.filters = filters
        self.attention = MultiHeadAttention(filters, filters, 1, residual=False, projection=True)
        # self.fc_control = Dense(filters * 3, final_filters, bnunits=final_filters)
        self.fc_control = Dense(filters * 3, final_filters, bnunits=final_filters)

    def forward(self, x, e, control_flag):
        x_controled = (x * control_flag).sum(axis=1, keepdim=True)
        e_controled = (e * control_flag).sum(axis=1, keepdim=True)

        h = self.attention(x_controled, x)

        h = paddle.concat([x_controled, e_controled, h], axis=2).reshape((x.shape[0], -1))
        # h = torch.cat([h, cnn_h.view(cnn_h.size(0), -1)], dim=1)
        h = self.fc_control(h)
        return h


class Dense(nn.Layer):
    def __init__(self, units0, units1, bnunits=0, bias=True):
        super().__init__()
        if bnunits > 0:
            bias = False
        self.dense = nn.Linear(units0, units1, bias_attr=bias)
        self.bnunits = bnunits
        self.bn = nn.BatchNorm1D(bnunits) if bnunits > 0 else None

    def forward(self, x):
        h = self.dense(x)
        if self.bn is not None:
            size = h.shape
            h = h.reshape((-1, self.bnunits))
            h = self.bn(h)
            h = h.reshape(size)
        return h


class CNNModel(nn.Layer):
    def __init__(self, final_filters):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(53, 128, kernel_size=1, stride=1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(128, 160, kernel_size=1, stride=1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(160, 128, kernel_size=1, stride=1, bias_attr=False),
            nn.ReLU()
        )
        self.pool1 = nn.AdaptiveAvgPool2D((1, 11))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2D(128),
            nn.Conv2D(128, 160, kernel_size=(1, 1), stride=1, bias_attr=False),
            nn.ReLU(),
            nn.BatchNorm2D(160),
            nn.Conv2D(160, 96, kernel_size=(1, 1), stride=1, bias_attr=False),
            nn.ReLU(),
            nn.BatchNorm2D(96),
            nn.Conv2D(96, final_filters, kernel_size=(1, 1), stride=1, bias_attr=False),
            nn.ReLU(),
            nn.BatchNorm2D(final_filters),
        )
        self.pool2 = nn.AdaptiveAvgPool2D((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x['cnn_feature']
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        return x


class ActionHistoryEncoder(nn.Layer):
    def __init__(self, input_size=19, hidden_size=64, num_layers=2, bidirectional=True):
        super().__init__()
        self.action_emd = nn.Embedding(19, 8)
        self.rnn = nn.GRU(8, hidden_size, num_layers, time_major=False, direction="bidirect")

    def forward(self, x):
        h = self.action_emd(x['action_history'])
        h = h.squeeze(axis=2)
        h, _ = self.rnn(h)
        return h


class FootballHead(nn.Layer):
    def __init__(self, filters):
        super().__init__()
        self.head_p = nn.Linear(filters, 19, bias_attr=False)
        #self.head_p_special = nn.Linear(filters, 1 + 8 * 4, bias=False)
        #self.head_v = nn.Linear(filters, 1, bias=True)
        self.head_r = nn.Linear(filters, 1, bias_attr=False)

    def forward(self, x):
        p = self.head_p(x)
        #p2 = self.head_p_special(x)
        #v = self.head_v(x)
        r = self.head_r(x)
        return p, r#orch.cat([p, p2], -1), v, r
#%%


if __name__ == "__main__":
    from football_env import FootballEnv
    import football_model as torch_model

    env = gfootball_env.create_environment(env_name="11_vs_11_kaggle",
                                           representation="raw",
                                           rewards="scoring,checkpoints")
    env = FootballEnv(env)
    obs = env.reset()
    obs = to_tensor(obs, unsqueeze=0)
    net = FootballNet()
    s = net(obs)
    state_dict = net.state_dict()
    torch_net = torch_model.FootballNet()
    state_dict2 = torch_net.state_dict()

    print("...")






