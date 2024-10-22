import os
import time
import random
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertPreTrainedModel, BertModel
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

MAX_LEN = 128
seed_val = 0

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

labels_encoding = {
    "finance": 0,
    "entertainment": 1,
    "sports": 2,
    "news": 3,
    "autos": 4,
    "video": 5,
    "lifestyle": 6,
    "travel": 7,
    "health": 8,
    "foodanddrink": 9,
}

def read_tsv(file_path: str, skip_first=False):
    data = []
    with open(file_path, "r") as fin:
        for idx, line in enumerate(fin):
            if skip_first and not idx:
                continue
            skip_flag = False
            segments = line.strip().split("\t")
            assert len(segments) == 3, segments
            if not skip_flag:
                data.append({
                    "sentence": segments[0] + segments[1], # combine news title with news body
                    "label": segments[2],
                })
    return pd.DataFrame(data)

def gather_data():
    num_for_train = 8000
    num_for_dev_test = 1000
    save_path = "splits_data"
    datasets = []
    lang_ids = ["de", "en", "es", "fr", "ru"]
    for lang_id in lang_ids:
        dev = read_tsv("xglue_full_dataset/NC/xglue.nc.{}.dev".format(lang_id)) # test doesn't have labels / no train for other langs
        datasets.append((lang_id, dev))

    # downsample training sets to simulate FL scenario
    for (lang_id, dev) in datasets:
        print(lang_id, "saving to file")
        save_path = f"nc/{lang_id}"
        if not os.path.isdir("nc"):
            os.makedirs("nc")

        all_data = dev.sample(frac=1)
        train_sampled = all_data.iloc[:num_for_train]
        dev = all_data.iloc[num_for_train:num_for_train+num_for_dev_test]
        test = all_data.iloc[num_for_train+num_for_dev_test : ]
        dev.to_csv(save_path + "_dev.csv", index=None)
        test.to_csv(save_path + "_test.csv", index=None)
        train_sampled.to_csv(save_path + "_train.csv", index=None)

        print(f"train_sampled shape {train_sampled.shape}")
        print(f"dev shape {dev.shape}")
        print(f"test shape {test.shape}")

    return {}

def preprocess(df):
    sentences = df.sentence.values[1:]
    labels = np.array([labels_encoding[l] for l in df.label.values[1:]])
    tokenizer = BertTokenizer.from_pretrained('bert_pretrain/vocab.txt')
    
    encoded_sentences = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(
                            sent,
                            add_special_tokens = True,
                            truncation=True,
                            max_length = MAX_LEN
                    )
        encoded_sentences.append(encoded_sent)
    encoded_sentences = pad_sequences(encoded_sentences, maxlen=MAX_LEN, dtype="long", 
                            value=0, truncating="post", padding="post")
    return encoded_sentences, labels

def attention_masks(encoded_sentences):
    # attention masks, 0 for padding, 1 for actual token
    attention_masks = []
    for sent in encoded_sentences:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks

if not os.path.exists('nc/de_dev.csv'):
    gather_data()

for domain in ['de', 'en', 'es', 'fr', 'ru']:
    path1, path2 = 'nc/{}_train.csv'.format(domain), 'nc/{}_test.csv'.format(domain)
    df1 = pd.read_csv(path1, delimiter=',', header=None, names=['sentence', 'label'])
    df2 = pd.read_csv(path2, delimiter=',', header=None, names=['sentence', 'label'])
    encoded_sentences1, labels1 = preprocess(df1)
    encoded_sentences2, labels2 = preprocess(df2)
    attention_mask1 = attention_masks(encoded_sentences1)
    attention_mask2 = attention_masks(encoded_sentences2)
    inputs1, labels1, masks1 = np.array(encoded_sentences1), np.array(labels1), np.array(attention_mask1)
    inputs2, labels2, masks2 = np.array(encoded_sentences2), np.array(labels2), np.array(attention_mask2)
    print('Domain: {}\tCloud: {}\tDevice: {}'.format(domain, inputs1.shape, inputs2.shape))
    np.save('{}_cloud.npy'.format(domain), {'inputs': inputs1, 'masks':masks1, 'labels':labels1})
    np.save('{}_device.npy'.format(domain), {'inputs': inputs2, 'masks': masks2, 'labels': labels2})


