MAX_WORDS = 20 # 单句最长单词数
MIN_COUNT = 6 # 单词最低词频

PAD_TOKEN = 0 # 填充标记
SOS_TOKEN = 1 # 开始标记<start>
EOS_TOKEN = 2 # 结束标记<end>

DATASET_PATH = './datasets'

# for training
attn_model = 'dot' #注意力机制
hidden_size = 500 # 隐含层单元数
encoder_n_layer = 2 # 编码器层数
decoder_n_layer = 2 # 解码器层数
dropout = 0.1
clip = 50.0 # 梯度裁剪阈值
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decay_rate = 0.1
decoder_learning_ratio = 5.0
n_iteration = 4500 #迭代次数
batch_size = 32 # 批次大小
print_every = 10 # 打印周期
save_every = 500 #保存周期
model_name = 'chatbot'
dataset_name = 'cornell'
# dataset_name = 'cornell'
encoder_n_layers = 2
decoder_n_layers = 2
save_dir = './model/'
loadFilename = None

