import torch
import torch.nn as nn
import torch.nn.functional as F


# 编码器
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = embedding
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)  # 双向gru

    def forward(self, input_seq, input_lengths, hidden=None):
        '''
        :param input_seq: 一组输入语句,shape=(max_Length,batch_size)
        :param input_lengths: 与输入语句对应的各语句长度
        :param hidden: 隐含层单元数
        :return: outputs:the output features ht from the last layer of the GRu,for each timestep (stum of bidirectional outputs)
                 hidden:hidden state for the Last timestep,of shape=(n_Layers x num directions,batch size,hidden size)
        '''

        # 注意，一个embedding层用于在任意大小的特征空间中编码我们的单词索引。对于我们的模型，此图层会将每个单词映射到大小为hidden_​​size的要素空间。训练后，这些值应编码相似意义词之间的语义相似性。
        # 最后，如果将填充的一批序列传递给RNN模块，我们必须分别使用torch.nn.utils.rnn.pack_padded_sequence和torch.nn.utils.rnn.pad_packed_sequence。
        # 来打包和解包RNN传递周围的填充
        embedded = self.embedding(input_seq)
        # print('embedded:')
        # print(embedded.shape)
        # 去填充， 打包成单一列表
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # 经过RNN
        outputs, hidden = self.gru(packed, hidden)
        # 填充已打包的列表
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # 总和双向rnn
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # sum bidirectionl
        return outputs, hidden


# 注意力机制
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

    def dot_score(self, decoder_output, encoder_output):
        return torch.sum(decoder_output * encoder_output, dim=2)

    def forward(self, decoder_output, encoder_outputs):
        # hidden of shape:(1,batch size,hidden size)
        # encoder outputs of shape:(max Length,batch size,hidden size)
        # (1,batch size,hidden size)*(max Length,batch size,hidden size)=(max length,batch size,hidden size)
        attn_energies = self.dot_score(decoder_output, encoder_outputs)  # (max_length, batch_size)
        attn_energies = attn_energies.t()  # (batch_size, max_length)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # (batch_size, 1, max_length)


class DecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.attn = Attn(attn_model, hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # input_step:one time step (one word) of input sequence batch;shape=(1,batch_size)
        # st_hidden:final hidden state of GRU; shape=(n_layers x num_directions,batch_size,hidden_size)
        # encoder outputs:encoder model's output;shape=(max_length,batch_size,hidden_size)
        # Note:we run this one step(word)at a time
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        rnn_output, hidden = self.gru(embedded, last_hidden)
        # rnn output of shape:(1,batch,num directions *hidden size)
        # hidden of shape:(num Layers *num directions,batch,hidden size)

        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum"context vector
        # (batch size,1,max length)bmm with (bacth size,max Length,hidden)=(batch size,1,hidden)

        # 组乘法
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden
        # output: softmax normalized tensor giving probabilities of each word being the correct next word in the decoded sequence
        # shape=(batch size, voc. num words)
        # hidden: final hidden state of GRU; shape=(n layers x num directions, batch size, hidden size)