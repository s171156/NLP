import time
import pandas as pd
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
# Tokenizerの準備
from transformers import BertJapaneseTokenizer
from my_module import path_manager as pm
import torch
import numpy as np
import statistics
import sys
# シードを固定することで学習の再現性を担保する。
torch.manual_seed(0)

# 学習済みモデルのロード
model_name = str(pm.get_abs_path(
    __file__, 'resources/BERT/BERT-base_mecab-ipadic-bpe-32k_whole-word-mask'))
# Tokenizerのセット
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
# CPU/GPU環境の設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 学習に利用するセンテンスを決定する境界値
boundary = 0.2
# ラベル付きデータ（極性値を付与したレビューのセンテンス）の読み込み
df = pd.read_csv('sent_senti.csv')
# 無極性のレビューを除去する。
df = df[df['score'] != 0]
# 感情値の絶対値が境界値以上の行を抽出
df = df[abs(df['score']) >= boundary]
# 感情値が正の場合は１、負の場合は0のラベルを付与した列を挿入
df['label'] = df['score'].apply(lambda x: 1 if 0 < x else 0)
# 文章中の単語数を集計した列を挿入
df['length'] = df['content'].apply(lambda x: len(tokenizer.tokenize(x)))
# 単語数を基準に降順にソート
df.sort_values('length', ascending=False, inplace=True)
# ネガティブな文章のデータフレーム
df_neg = df[df['label'] == 0]
# ポジティブな文章のデータフレーム
df_pos = df[df['label'] == 1]
# データ数の少ない一方の極性にデータ数を揃える。
if len(df_pos) > len(df_neg):
    df_pos = df_pos[:len(df_neg)]
elif len(df_neg) > len(df_pos):
    df_neg = df_neg[:len(df_pos)]
# データフレームを結合
df = pd.concat([df_pos, df_neg])
# 文章のリストを格納
sentences = df['content'].values
# ラベルのリストを格納
labels = df['label'].values


def show_status_of_sentences(boundary: float = 0.0):
    """
    センテンスの状態を表示します。
    """
    # # データフレームの読み込み
    # df = pd.read_csv('sent_senti.csv')
    # # 無極性のレビューを除去
    # df = df[df['score'] != 0]
    # # 感情値の絶対値が境界値以上の行を抽出
    # df = df[abs(df['score']) >= boundary]

    # ポジティブなレビューとネガティブなレビューの総数を取得
    pos = (df['score'] >= boundary).sum()
    neg = (df['score'] <= -boundary).sum()
    print(f'positive: {pos}, negative: {neg}')
    # センテンスの多い極性とセンテンスの総数の差を表示
    polarity = 'positive' if pos > neg else 'negative'
    diff = abs(pos - neg)
    print(f'polarity: {polarity}, difference: {diff}')

    # レビューの単語数を取得
    sentences = df['content'].values
    words = [len(tokenizer.tokenize(sent)) for sent in sentences]

    print(f'最大単語数: {max(words)}')
    print(f'最小単語数: {min(words)}')
    print(f'平均単語数: {round(statistics.mean(words))}')
    print(f'中央値: {statistics.median(words)}')
    print(f'最瀕値: {statistics.mode(words)}')

    print('上記の最大単語数にSpecial token（[CLS], [SEP]）の+2をした値が最大単語数')


# show_status_of_sentences()
# sys.exit()

input_ids = []
attention_masks = []

# 1文づつ処理
for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,  # Special Tokenの追加
        max_length=28,           # 文章の長さを固定（Padding/Trancatinating）
        pad_to_max_length=True,  # PADDINGで埋める
        return_attention_mask=True,   # Attention maksの作成
        return_tensors='pt',  # Pytorch tensorsで返す
    )

    # 単語IDを取得
    input_ids.append(encoded_dict['input_ids'])

    # Attention　maskの取得
    attention_masks.append(encoded_dict['attention_mask'])

# リストに入ったtensorを縦方向（dim=0）へ結合
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# tenosor型に変換
labels = torch.tensor(labels)

# # 確認
# print('Original: ', sentences[0])
# print('Token IDs:', input_ids[0])


# データセットクラスの作成
dataset = TensorDataset(input_ids, attention_masks, labels)


# 90%地点のIDを取得
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# データセットを分割
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('訓練データ数：{}'.format(train_size))
print('検証データ数:　{} '.format(val_size))

# データローダーの作成
batch_size = 32

# 訓練データローダー
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),  # ランダムにデータを取得してバッチ化
    batch_size=batch_size
)

# 検証データローダー
validation_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),  # 順番にデータを取得してバッチ化
    batch_size=batch_size
)

# BertForSequenceClassification 学習済みモデルのロード
model = BertForSequenceClassification.from_pretrained(
    model_name,  # 日本語Pre trainedモデルの指定
    num_labels=2,  # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
    output_attentions=False,  # アテンションベクトルを出力するか
    output_hidden_states=False,  # 隠れ層を出力するか
)

# モデルをGPUへ転送
model.cuda()


# 最適化手法の設定
optimizer = AdamW(model.parameters(), lr=2e-5)


# 訓練パートの定義
def train(model):
    model.train()  # 訓練モードで実行
    train_loss = 0
    for batch in train_dataloader:  # train_dataloaderはword_id, mask, labelを出力する点に注意
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        optimizer.zero_grad()
        loss, logits = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    return train_loss

# テストパートの定義


def validation(model):
    model.eval()  # 訓練モードをオフ
    val_loss = 0
    with torch.no_grad():  # 勾配を計算しない
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():
                (loss, logits) = model(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
            val_loss += loss.item()
    return val_loss


# 学習の実行
# max_epoch = 4
max_epoch = 5
train_loss_ = []
test_loss_ = []

for epoch in range(max_epoch):
    print('学習を開始しました')
    start = time.time()
    train_ = train(model)
    elapsed_time = time.time() - start
    print(f'学習に要した時間_1：{elapsed_time}')
    test_ = validation(model)
    elapsed_time = time.time() - start
    print(f'学習に要した時間_2：{elapsed_time}')
    train_loss_.append(train_)
    test_loss_.append(test_)


model.eval()  # 訓練モードをオフ

# ファインチューニング済みモデルのセーブ
# model.save_pretrained('my_pretrained_model')
model.save_pretrained(f'my_pretrained_models/undersampling/model_e{max_epoch}')

# 検証方法の確認（1バッチ分で計算ロジックに確認）

df_list = []

for batch in validation_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    with torch.no_grad():
        # 学習済みモデルによる予測結果をpredsで取得
        preds = model(b_input_ids,
                      token_type_ids=None,
                      attention_mask=b_input_mask)

    # 比較しやすい様にpd.dataframeへ整形
    # pd.dataframeへ変換（GPUに乗っているTensorはgpu->cpu->numpy->dataframeと変換）
    logits_df = pd.DataFrame(preds[0].cpu().numpy(),
                             columns=['logit_0', 'logit_1'])
    # np.argmaxで大き方の値を取得
    pred_df = pd.DataFrame(
        np.argmax(preds[0].cpu().numpy(), axis=1), columns=['pred_label'])
    label_df = pd.DataFrame(b_labels.cpu().numpy(), columns=['true_label'])

    accuracy_df = pd.concat([logits_df, pred_df, label_df], axis=1)
    df_list.append(accuracy_df)

df_result = pd.concat(df_list, axis=0)
# df_result.to_csv('result.csv', index=False)
df_result.to_csv(
    f'my_pretrained_models/results/undersampling/model_e{max_epoch}.csv', index=False)
