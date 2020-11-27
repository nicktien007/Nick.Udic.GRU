
from gru import GRU
from service.preprocess_service import to_prepare
from utils import logging_utils
import logging as log

import torch
from torch import nn
from torchtext import data
from torchtext.vocab import Vectors

import torch
from torchtext import data

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

import time
import math

import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname="./font/TaipeiSansTCBeta-Bold.ttf")#為了顯示中文，載入一個中文字體

device = torch.device('cpu')

def main():
    # to_prepare("./dataset/Taipei_QA_new.txt")
    TEXT, train_iter, test_iter = load_train_data()
    # 定義超參
    vocab_size = len(TEXT.vocab)
    embedding_dim = 400
    hidden_dim = 400
    layer_dim = 1
    output_dim = 149
    epoch = 10

    gru_model = GRU(input_size=vocab_size,
                     embedding_dim=embedding_dim,
                     hidden_dim=hidden_dim,
                     num_layers=layer_dim,
                     output_dim=output_dim)
    print(gru_model)

    optimizer = optim.Adam(gru_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    gru_model.to(device)

    grumodel, train_process = train_model(gru_model, train_iter, test_iter,
                                          criterion, optimizer, num_epoch=epoch)

    torch.save(grumodel, "./trained_model/taipei_qa_GRU.pt")
    torch.save(TEXT.vocab, "./trained_model/vocab")

    plt.figure(figsize=[16, 8])
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all,
             "r.-", label="Train Loss")
    plt.plot(train_process.epoch, train_process.test_loss_all,
             "bs-", label="Test Loss")
    plt.xlabel("Epoch number", fontsize=13)
    plt.ylabel("Loss value", fontsize=13)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all,
             "r.-", label="Train Acc")
    plt.plot(train_process.epoch, train_process.test_acc_all,
             "bs-", label="Test Acc")
    plt.xlabel("Epoch number", fontsize=13)
    plt.ylabel("Acc value", fontsize=13)
    plt.legend()
    plt.show()

    eval_RNN(test_iter)


def train_model(model, traindata_loader, testdata_loader,
                criterion, optimizer, num_epoch=25):
    train_loss_all, train_acc_all = [], []
    test_loss_all, test_acc_all = [], []
    learn_rate = []
    since = time.time()

    # 設置等間隔學習率，使得學習率動態變化，每隔step_size個epoch，學習率縮小到原來的1/10
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)
    start = time.time()
    for epoch in range(num_epoch):
        learn_rate.append(scheduler.get_last_lr()[0])
        print("-" * 30)
        print("Epoch {}/{}, Lr:{}".format(epoch, num_epoch - 1, learn_rate[-1]))

        # 每個epoch分為訓練階段和驗證階段
        train_loss, train_corrects, train_num = 0, 0, 0
        test_loss, test_corrects, test_num = 0, 0, 0

        model.train()
        for step, batch in enumerate(traindata_loader):
            textdata, target = batch.cutword[0], batch.labelcode.view(-1)
            # 這裡的batch.text[0]代表詞向量，batch.text[1]代表這些詞向量在詞表中的索引index
            out = model(textdata)
            pre_lab = torch.argmax(out, 1)
            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target)
            train_corrects += torch.sum(pre_lab == target)
            train_num += len(target)
            if step % 20 == 0:
                log.info(f'[{time_since(start)}] Epoch {epoch} ' + "Train Loss: {:.4f} Train Acc:{:.4f}".format(
                    train_loss / train_num,
                    train_corrects.double().item() / train_num))

        # 計算一個epoch的平均損失值和精度值
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)

        log.info("====Epoch {} Train Loss: {:.4f} Train Acc{:.4f} ====".format(epoch, train_loss_all[-1],
                                                                          train_acc_all[-1]))

        scheduler.step()  # 更新學習率

        # 計算在測試集上的損失和準確率
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(testdata_loader):
                textdata, target = batch.cutword[0], batch.labelcode.view(-1)
                out = model(textdata)
                pre_lab = torch.argmax(out, 1)
                loss = criterion(out, target)
                test_loss += loss.item() * len(target)
                test_corrects += torch.sum(pre_lab == target)
                test_num += len(target)

        test_loss_all.append(test_loss / test_num)
        test_acc_all.append(test_corrects.double().item() / test_num)

        log.info("====Epoch {} Test Loss: {:.4f} Test Acc{:.4f}====".format(epoch, test_loss_all[-1],
                                                                      test_acc_all[-1]))

    train_process = pd.DataFrame(data={
        "epoch": range(num_epoch),
        "train_loss_all": train_loss_all,
        "train_acc_all": train_acc_all,
        "test_loss_all": test_loss_all,
        "test_acc_all": test_acc_all
    })
    return model, train_process


def eval_RNN(test_iter):
    rnn_model = torch.load("./trained_model/taipei_qa_GRU.pt")
    rnn_model.eval()  # 模式設為評估模式，梯度不再更新
    predict_labels = []
    true_labels = []

    for step, batch in enumerate(test_iter):
        textdata, target = batch.cutword[0], batch.labelcode.view(-1)
        out = rnn_model(textdata)
        pre_lab = torch.argmax(out, 1)
        predict_labels += pre_lab.flatten().tolist()
        true_labels += target.flatten().tolist()

    acc = accuracy_score(predict_labels, true_labels)
    print(f"在測試集上的精度為:{acc}")

    class_label = [str(i) for i in range(0, 149)]
    # 計算混淆矩陣並可視化
    conf_mat = confusion_matrix(predict_labels, true_labels)
    conf_mat_df = pd.DataFrame(conf_mat, index=class_label, columns=class_label)

    # 繪製熱力圖
    plt.subplots(figsize=(50, 45))

    heatmap = sns.heatmap(conf_mat_df, annot=True, fmt="d", cmap="hot_r")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
                                 ha="right", fontproperties=fonts)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
                                 ha="right", fontproperties=fonts)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

def load_train_data():
    TEXT = data.Field(sequential=True,  # 表明輸入的數據是文本數據
                      tokenize=lambda x: x.split(" "),  # 分詞邏輯
                      include_lengths=True,  # 包含字符長度的信息
                      use_vocab=True,  # 建立詞表
                      batch_first=True,  # batch優先的數據方式
                      fix_length=400  # 每個句子固定長度為400
                      )

    LABEL = data.Field(sequential=True,
                       tokenize=lambda x: [int(x)],  # 只有指明sequential=True， tokenize才會被執行
                       use_vocab=False,  # 不創建詞表
                       pad_token=None,  # 不進行填充
                       unk_token=None  # 沒有無法識別的字符
                       )

    text_data_field = [
        ("labelcode", LABEL),
        ("cutword", TEXT)
    ]
    # 通過上述方式定義讀取數據的邏輯

    traindata, testdata = data.TabularDataset.splits(
        path= "./train_data", format="csv",
        train="qa_balance_train.csv", fields=text_data_field,
        test="qa_balance_test.csv",
        skip_header=True
    )
    print(len(traindata), len(testdata))
    # print
    for item in traindata:
        print(item.cutword)
        print(item.labelcode)
        break

    vectors = Vectors(
        name= "./trained_model/word2vec_wiki_zh.model.txt"
    )

    # 建立詞表
    TEXT.build_vocab(traindata, max_size=20000, vectors=vectors)
    LABEL.build_vocab(traindata)

    # 可視化前25個高頻詞
    word_fre = TEXT.vocab.freqs.most_common(n=25)
    word_fre = pd.DataFrame(data=word_fre, columns=["word", "fre"])

    plt.figure(dpi=300)
    word_fre.plot(x="word", y="fre", kind="bar", legend=False, figsize=[20, 7])
    plt.xticks(rotation=0, fontproperties=fonts, size=15)
    # plt.grid(True)
    plt.show()

    # 定義數據加載器
    BATCH_SIZE = 32
    train_iter = data.BucketIterator(traindata, batch_size=BATCH_SIZE, device=device)
    test_iter = data.BucketIterator(testdata, batch_size=BATCH_SIZE, device=device)

    for batch in train_iter:
        text, pos = batch.cutword
        label = batch.labelcode
        print("text.shape:", text.shape)
        print("pos.shape:", pos.shape)
        print("label.shape:", label.shape)
        print("第一句話前10個字", text[0][:10])
        break

    # 可視化Label
    word_fre = LABEL.vocab.freqs.most_common()
    # print(word_fre)
    word_fre = pd.DataFrame(data=word_fre, columns=["x", "y"])

    plt.figure(dpi=300)
    word_fre.plot(x="x", y="y", kind="bar", legend=False, figsize=[50, 20])
    plt.xticks(rotation=90, fontproperties=fonts, size=15)
    # plt.grid(True)
    plt.show()

    return TEXT, train_iter, test_iter

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

if __name__ == '__main__':
    logging_utils.Init_logging()
    main()