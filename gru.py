from torch import nn


class GRU(nn.Module):
    """
    input_size = 400  # input的維度
    hidden_dim = 400  # 隱藏層維度
    num_layers = 1    # GRU迭代次數
    label_num = 149   # 總Label數量
    max_len = max_len # 句子最大長度->60
    batch_size = 1    # batch_size
    """

    def __init__(self, input_size, embedding_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_size = input_size

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.5)  # input_size,  隱藏層維度
        self.fc = nn.Linear(hidden_dim, output_dim)  # 將句向量經過一層liner判斷類別

    def forward(self, x):
        # x shape (batch, time_step, input_size), time_step--->seq_len
        # r_out shape (batch, time_step, output_size), out_put_size--->num_directions*hidden_size
        # h_0 shape (num_layers*num_directions, batch, hidden_size), here we use zero initialization
        # h_n shape (num_layers*num_directions, batch, hidden_size)
        embeds = self.embedding(x)

        # initialization hidden state
        # 1.zero init
        # None represents zero initial hidden state
        r_out, h_n = self.gru(embeds, None)

        # [B, hidden_size*num_directions]
        out = self.fc(r_out[:, -1, :])

        return out
