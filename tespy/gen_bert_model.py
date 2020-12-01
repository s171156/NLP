import pathlib
import transformers
from my_module import path_manager as pm
import tensorflow as tf
import numpy as np
import pandas as pd
import statistics

model_name = str(pm.get_abs_path(
    __file__, 'resources/BERT/BERT-base_mecab-ipadic-bpe-32k'))
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)


# テキストのリストをtransformers用の入力データに変換
def to_features(texts, max_length):
    shape = (len(texts), max_length)
    # input_idsやattention_mask, token_type_idsの説明はglossaryに記載(cf. https://huggingface.co/transformers/glossary.html)
    input_ids = np.zeros(shape, dtype="int32")
    attention_mask = np.zeros(shape, dtype="int32")
    token_type_ids = np.zeros(shape, dtype="int32")
    for i, text in enumerate(texts):
        encoded_dict = tokenizer.encode_plus(
            text, max_length=max_length, pad_to_max_length=True)
        input_ids[i] = encoded_dict["input_ids"]
        attention_mask[i] = encoded_dict["attention_mask"]
        token_type_ids[i] = encoded_dict["token_type_ids"]
    return [input_ids, attention_mask, token_type_ids]


# 単一テキストをクラス分類するモデルの構築
def build_model(model_name, num_classes, max_length):
    input_shape = (max_length, )
    input_ids = tf.keras.layers.Input(input_shape, dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(input_shape, dtype=tf.int32)
    token_type_ids = tf.keras.layers.Input(input_shape, dtype=tf.int32)
    bert_model = transformers.TFBertModel.from_pretrained(model_name)
    last_hidden_state, pooler_output = bert_model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )
    output = tf.keras.layers.Dense(
        num_classes, activation="softmax")(pooler_output)
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids], outputs=[output])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy", metrics=["acc"])
    return model


def train(train_texts, train_labels, model_name):

    num_classes = 2
    # max_length = 15
    max_length = 27
    batch_size = 10
    epochs = 3

    x_train = to_features(train_texts, max_length)
    y_train = tf.keras.utils.to_categorical(
        train_labels, num_classes=num_classes)
    model = build_model(model_name, num_classes=num_classes,
                        max_length=max_length)

    # 訓練
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
    )

    model.save_weights('resources/train/checkpoints/my_checkpoint')


def gen_model(boundary: float = 0.0):
    '''
    BERTモデルを生成します。

    Parameters
    ---
    boundary (float, optional): 学習に用いる極性値を決定する境界値
    '''
    # データフレームの読み込み
    df = pd.read_csv('sent_senti.csv')
    # 無極性のレビューを除去する。
    df = df[df['score'] != 0]
    # 感情値の絶対値が境界値以上の行を抽出
    df = df[abs(df['score']) >= boundary]
    train_texts = df['content'].values
    # 感情値が正の場合は１、負の場合は0
    train_labels = [1 if 0 < score else 0 for score in df['score'].values]
    # 学習を開始
    train(train_texts, train_labels, model_name)


def load_model():
    num_classes = 2
    # max_length = 15
    max_length = 27
    model = build_model(model_name, num_classes=num_classes,
                        max_length=max_length)
    model.load_weights('resources/train/checkpoints/my_checkpoint')
    return model


if __name__ == "__main__":
    # show_status_of_sentences()
    # gen_model(boundary=0.2)
    # model = load_model()
    # df = pd.read_csv('sent_senti.csv')
    # df = df[df['score'] != 0]
    # test_texts = df['content'].values[:100]
    # test_labels = [1 if 0 < score else -
    #                1 for score in df['score'].values][:100]
    # max_length = 27
    # # 予測
    # x_test = to_features(test_texts, max_length)
    # y_test = np.asarray(test_labels)
    # y_preda = model.predict(x_test)
    # y_pred = np.argmax(y_preda, axis=1)
    # print("Accuracy: %.5f" % accuracy_score(y_test, y_pred))
    # model.evaluate()
    pass
