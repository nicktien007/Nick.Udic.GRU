import numpy as np
import pandas as pd
from udicOpenData.stopwords import rmsw


def to_prepare(path):
    """資料預處理"""
    train_df = pd.read_csv(path, sep=" ")

    labels = list(train_df["label"])
    label_map = dict()
    label_idx = 0
    # one hot encoder
    for label in labels:
        if label not in label_map:
            label_map[label] = label_idx
            label_idx += 1

    print(len(label_map.keys()))

    # 調用rmsw分詞, 通過空格將處理後的詞語拼接成字串
    train_df["cutword"] = train_df.text.apply(lambda t: " ".join(rmsw(t)))
    train_df["labelcode"] = train_df["label"].map(label_map)

    return to_balance(train_df[["labelcode", "cutword"]])


def to_balance(pd_all):
    """構造平衡語料"""

    print('數目（全）：%d' % pd_all.shape[0])

    label_list = []
    for i in range(149):
        label_list.append(pd_all[pd_all.labelcode == i])

    # sample_size = 8195 // 149  = 55
    sample_size = 55
    new_list = []

    for i, label in enumerate(label_list):
        new_list.append(label.sample(sample_size, replace=label.shape[0] < sample_size))

    # print(new_list)
    pd_corpus_balance = pd.concat(new_list)
    # print(pd_corpus_balance)

    # 先將數據分為train.csv和test.csv
    split_dataFrame(df=pd_corpus_balance,
                    train_file='./train_data/qa_balance_train.csv',
                    val_testfile='./train_data/qa_balance_test.csv',
                    seed=788,
                    ratio=0.1)


def split_dataFrame(df, train_file, val_testfile, seed=999, ratio=0.2):
    idxs = np.arange(df.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idxs)
    val_size = int(len(idxs) * ratio)
    df.iloc[idxs[:val_size], :].to_csv(val_testfile, index=False)
    df.iloc[idxs[val_size:], :].to_csv(train_file, index=False)
