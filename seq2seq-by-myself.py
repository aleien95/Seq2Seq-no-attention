# coding = 'utf-8'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from langconv import *
from tqdm import tqdm
import numpy as np
import os
import random
import time
# 数据预处理
def read_data(config):
    with open(config.data_address, 'r', encoding='utf-8') as f:
        data = f.read()
    line_data = data.split('\n')
    englist_text, chinese_text = [], []
    for text in line_data:
        text.split('\t')
        englist_text.append(text.split('\t')[0].lower())
        chinese_text.append(wiki2ch(text.split('\t')[1]))
    return englist_text, chinese_text

# 繁体转简体
def wiki2ch(sen):
    sentence = Converter('zh-hans').convert(sen)
    return sentence

# 构造字典
def bulid_dic(text_data):
    word2id = {}
    index = 4
    # 基本字典
    basic_dict = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
    for sen in text_data:
        for s in range(len(sen)):
            if word2id.get(sen[s]) == None:
                word2id[sen[s]] = index
                index += 1
    word2id.update(basic_dict)
    id2word = {v:k for k, v in word2id.items()}
    return word2id, id2word
def save_all_dic(en2id, ch2id):
    save_en2id = np.savetxt('en2id.txt', en2id)
    save_ch2id = np.savetxt('ch2id.txt', ch2id)
    return 0

def detect_dict_exists(en_text, ch_text):
    if os.path.exists('en2id.txt') and os.path.exists('ch2id.txt'):
        en_word2id = np.loadtxt('en2id.txt')
        ch_word2id = np.loadtxt('ch2id.txt')
    else:
        # 构造字典
        en_word2id, en_id2word = bulid_dic(en_text)
        ch_word2id, ch_id2word = bulid_dic(ch_text)
    return en_word2id, ch_word2id

# 将句子映射
def encoding_text(text_list, word2id):
    id_data = []
    for text in text_list:
        temp_list = []
        for i in range(len(text)):
            temp_list.append(word2id[text[i]])
        temp_list.append(3)
        id_data.append(temp_list)
    return id_data

class Translate_data(Dataset):
    def __init__(self, input_data, target_data):
        super(Translate_data, self).__init__()
        self.input_data = input_data
        self.target_data = target_data
    def __len__(self):
        return len(self.input_data)
    def __getitem__(self, item):
        src_sample = self.input_data[item]
        src_sample_len = len(self.input_data[item])
        src_target = self.target_data[item]
        src_target_len = len(self.target_data[item])
        return {'src':src_sample, 'src_len':src_sample_len, 'tar':src_target, 'tar_len':src_target_len}

def padding_batch(batch):
    src_len = [d['src_len'] for d in batch]
    tar_len = [d['tar_len'] for d in batch]
    max_src = max(src_len)
    max_tar = max(tar_len)

    for d in batch:
        d['src'].extend([0] * (max_src - d['src_len']))
        d['tar'].extend([0] * (max_tar - d['tar_len']))
    src = torch.tensor([pair['src'] for pair in batch], dtype = torch.long, device='cuda')
    tar = torch.tensor([pair['tar'] for pair in batch], dtype = torch.long, device='cuda')
    return {'src':src.t(), 'src_len':src_len, 'tar':tar.t(), 'tar_len':tar_len}

# 构造encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, encoder_drop_out):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_div = embedding_dim
        self.hidden = hidden_dim
        self.num_layers = num_layers
        self.drop_out = encoder_drop_out
        self.embedding = nn.Embedding(self.input_dim, self.embedding_div)
        self.gru = nn.LSTM(self.embedding_div, self.hidden, self.num_layers, dropout=self.drop_out, bidirectional=True)
    def forward(self, input, input_len, hidden):
        embeded = self.embedding(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeded, input_len, enforce_sorted=False)
        lstm_out, hidden = self.gru(packed, hidden)
        outputs, output_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, bidirectional):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(self.output_dim, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(2 * hidden_dim, self.output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        # input: torch.Size([32])
        embedded = self.embedding(input)
        # embedding: torch.Size([32, 256])
        out = self.dropout(embedded.view(1, batch_size, -1))
        # resize embedded: torch.Size([1, 32, 256])
        out, hidden = self.lstm(out, hidden)
        # lstm output: torch.Size([1, 32, 1024]) <class 'torch.Tensor'>
        out = self.fc(out.squeeze(0))
        out = self.softmax(out)
        # after softmax: <class 'torch.Tensor'>
        return out, hidden

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, ch_vocab_len, device, teacher_force):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_force = teacher_force
        self.ch_vocab = ch_vocab_len
        self.device = device
        self.predict = False
        self.max_len = 100

    def forward(self, input, encoder_len, target, target_len, aaaa=0.5):
        # 获取batch_size
        batch_size = input.size(1)
        # 定义hidden，[num_layers * bidirect, batch_size, hidden_size]
        # hidden_init = torch.zeros(config.n_layers * 2 if config.bidirectional else 1, 1, config.hidden_dim, device=config.device)
        state_h = torch.zeros(config.n_layers * 2, batch_size, config.hidden_dim).to(
            config.device)  # 起始的hidden status
        state_c = torch.zeros(config.n_layers * 2, batch_size, config.hidden_dim).to(
            config.device)  # 起始的cell status
        hidden_init = (state_h, state_c)
        out, encoder_out = self.encoder(input, encoder_len, hidden_init)
        # decoder的隐藏层初始化
        decoder_hidden = encoder_out
        # 定义decoder初始化输入
        decoder_input = torch.tensor([0] * batch_size, dtype=torch.long, device=config.device)
        if self.predict:
            # 预测阶段使用
            # 一次只输入一句话
            assert batch_size == 1, "batch_size of predict phase must be 1!"
            output_tokens = []

            while True:
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden
                )
                # [1, 1]
                # topk返回元组，第一个为值，第二个为下标
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(1)  # 上一个预测作为下一个输入
                # detach是截断求导，就是说这个位置的参数就不用求导了
                # item用来得到元素值
                output_token = topi.squeeze().detach().item()
                if output_token == 3 or len(output_tokens) == self.max_len:
                    break
                output_tokens.append(output_token)
            return output_tokens
        else:
            # 获取最大句子长度，用于整体截断
            max_length = max(target_len)
            # 保存输出每个词的id
            out_result_save = torch.zeros(max_length, batch_size, self.ch_vocab).to(self.device)
            for i in range(max_length):
                teacher_forcing = True if random.random() < aaaa else False
                out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                if teacher_forcing:
                    # 下一个输入来自target
                    decoder_input = target[i]
                else:
                    # 下一个输入来自上一个的预测值
                    value, indexs = out.topk(1)
                    decoder_input = indexs.squeeze(1)
                out_result_save[i] = out
            # # 计算损失,topk默认最后一维
            #
            # print(value.size(), '\n', indexs.size())
            # print('target size:', target.size())
            # 计算损失
            loss_fn = nn.NLLLoss(ignore_index=0)
            # out_result_save[sen_max_len, batch_size, output_dim]->[sen_max_len * batch_size, output_dim]
            # target[sen_max_len, batch_size]
            loss = loss_fn(out_result_save.reshape(-1, self.ch_vocab), target.reshape(-1))
            return loss

#验证集
def evaluate(
    model,
    data_loader,
    print_every=None
    ):
    model.predict = False
    model.eval()
    if print_every == 0:
        print_every = 1

    print_loss_total = 0  # 每次打印都重置
    start = time.time()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):

            # shape = [seq_len, batch]
            input_batchs = batch["src"]
            target_batchs = batch["tar"]
            # list
            input_lens = batch["src_len"]
            target_lens = batch["tar_len"]

            loss = model(input_batchs, input_lens, target_batchs, target_lens, aaaa=0)
            print_loss_total += loss.item()
            epoch_loss += loss.item()

            if print_every and (i+1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('\tCurrent Loss: %.4f' % print_loss_avg)

    return epoch_loss / len(data_loader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 定义训练
def train(seq2seq,
        train_data_loader,
        optimizer,
        clip=1,
        teacher_forcing_ratio=0.5,
        print_every=None):
    seq2seq.predict = False
    seq2seq.train()
    epoch_loss = 0
    print_every = 1
    print_loss_total = 0
    for i, data in tqdm(enumerate(train_data_loader)):
        optimizer.zero_grad()
        src = data['src']
        src_len = data['src_len']
        tar = data['tar']
        tar_len = data['tar_len']
        loss = seq2seq(src, src_len, tar, tar_len)
        epoch_loss += loss.item()
        print_loss_total += loss.item()
        # print('loss:', loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), 1)
        optimizer.step()
        if print_every and (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('\tCurrent Loss: %.4f' % print_loss_avg)
    return epoch_loss / len(train_data_loader)


def translate(
    model,
    sample,
    idx2token=None
    ):
    model.predict = True
    model.eval()

    # shape = [seq_len, 1]
    input_batch = sample["src"]
    # list
    input_len = sample["src_len"]

    output_tokens = model(input_batch, input_len, input_batch, input_len)
    output_tokens = [idx2token[t] for t in output_tokens]

    return "".join(output_tokens)


# 定义参数
class Conifg():
    def __init__(self):
        self.data_address = './cmn.txt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = 0
        self.output_dim = 0
        self.batch_size = 32
        self.encoder_embedding_dim = 256
        self.decoder_embedding_dim = 256
        self.bidirectional = True
        self.n_layers = 2
        self.n_epochs = 1
        self.clip = 1
        self.learning_rate = 1e-4
        self.encoder_drop_out = 0.5
        self.decoder_drop_out = 0.5
        self.hidden_dim = 512
        self.dict_address = None

if __name__ == '__main__':
    # 全局参数
    config = Conifg()
    en_text, ch_text = read_data(config)
    # en_word2id, ch_word2id = detect_dict_exists(en_text, ch_text)
    en_word2id, en_id2word = bulid_dic(en_text)
    ch_word2id, ch_id2word = bulid_dic(ch_text)
    # 证明每次创建的字典都是相同的，如果用了set，那么由于没有字典保存，模型就不没有迁移能力了
    # save = save_all_dic(en_word2id, ch_word2id)
    print('en_word2id', en_word2id)
    print('ch_word2id', ch_word2id)
    # 更新语料库长度
    config.input_dim = len(en_word2id)
    config.output_dim = len(ch_word2id)
    # 对数据进行编码
    input_data = encoding_text(en_text, en_word2id)
    target_data = encoding_text(ch_text, ch_word2id)
    # 验证是否句子编码的最后加上了<eos>
    # 发现自己写列表推导式的样子好帅哦
    print('input_data:', input_data[6])
    print('input_text:', [en_id2word[i] for i in input_data[6]])
    print('target_data:', target_data[6])
    print('target_text:', [ch_id2word[i] for i in target_data[6]])
    # 定义编码器
    encoder = Encoder(config.input_dim, config.encoder_embedding_dim, config.hidden_dim, config.n_layers, config.encoder_drop_out).to(config.device)
    # encoder = Encoder1(config.input_dim, config.encoder_embedding_dim, config.hidden_dim, config.n_layers).to(config.device)
    encoder.train()
    # 定义解码器
    decoder = Decoder(config.output_dim, config.decoder_embedding_dim, config.hidden_dim, config.n_layers, bidirectional=True).to(config.device)
    decoder.train()
    # 定义Seq2seq
    seq2seq = Seq2seq(encoder, decoder, config.output_dim, config.device, teacher_force=0.5)
    seq2seq.load_state_dict(torch.load('en2ch-model.pt'))
    # 定义优化器
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=config.learning_rate)
    # Dataset包装
    data = Translate_data(input_data, target_data)
    train_data_loader = DataLoader(data, batch_size=32, shuffle=True, collate_fn=padding_batch)
    # 构造训练集和验证集合dataset
    # train_size = int(0.9 * len(input_data))
    # test_size = len(input_data) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    # 构造训练和验证集dataloader
    # train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=padding_batch)
    # text_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=padding_batch)
    best_valid_loss = float('inf')
    for epoch in range(config.n_epochs):
        start_time = time.time()
        # 训练
        train_loss = train(seq2seq, train_data_loader, optimizer, 1, teacher_forcing_ratio=0.5, print_every=None)
        valid_loss = evaluate(seq2seq, train_data_loader)
        end_time = time.time()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(seq2seq.state_dict(), 'en2ch-model.pt')
        if epoch % 2 == 0:
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
    print("best valid loss：", best_valid_loss)
    # 加载最优权重
    seq2seq.load_state_dict(torch.load('en2ch-model.pt'))
    random.seed(2021)
    for i in random.sample(range(len(input_data)), 10):  # 随机看10个
        en_tokens = list(filter(lambda x: x!=0, input_data[i]))  # 过滤零
        ch_tokens = list(filter(lambda x: x!=3 and x!=0, target_data[i]))  # 和机器翻译作对照
        sentence = [en_id2word[t] for t in en_tokens]
        print("【原文】")
        print("".join(sentence))
        translation = [ch_id2word[t] for t in ch_tokens]
        print("【原文】")
        print("".join(translation))
        test_sample = {}
        test_sample["src"] = torch.tensor(en_tokens, dtype=torch.long, device=config.device).reshape(-1, 1)
        test_sample["src_len"] = [len(en_tokens)]
        print("【机器翻译】")
        print(translate(seq2seq, test_sample, ch_id2word), end="\n\n")