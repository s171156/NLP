import gensim
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import oseti
import random
from transformers import BertForSequenceClassification
import fasttext
import gensim
# model = fasttext.load_model('resources/fasttext_model/model_e20.bin')
# model = gensim.models.KeyedVectors.load_word2vec_format('hoge.bin')


def tes_oseti():
    analyzer = oseti.Analyzer()
    score = analyzer.analyze_detail('このイヤホンは重低音が良いです')
    print(score)


def test_tfidf(corpus: list):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    idf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    df_idf = pd.DataFrame(columns=['idf']).from_dict(
        dict(idf), orient='index')
    df_idf.columns = ['idf']
    print(df_idf.sort_values("idf", ascending=False).head(10).T)

    # df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    # print(df_tfidf)


def rename_df_col():
    df = pd.DataFrame({'A_a': [1, 2, 3], 'B_b': [1, 2, 3], 'C_c': [1, 2, 3]})
    df.rename(columns=lambda x: x.split('_')[1], inplace=True)
    # def columns(x): return {x.split('_')[0]: x.split('_')[1]}
    # print(columns(df.columns.values))
    print(df)


def extract_num():
    li = [round(random.random(), 1) for i in range(10)]
    # df = pd.DataFrame(li, columns=['f*ck'])
    # print(df)
    # df = df[abs(df['f*ck']) > 0.9]
    # print(df)
    # print(li)
    # li = [i if i >= 0.5 else -i for i in li if i > 0.2]
    # print(li)
    print(abs(li))


def gen_report(decimals: int = 6):
    epoch = 5
    prefix = ''
    df = pd.read_csv(
        f'my_pretrained_models/results/undersampling/model{prefix}_e{epoch}.csv')
    y_pred = df['pred_label'].values
    y_true = df['true_label'].values
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).round(decimals)
    print(df)
    df.to_csv(
        f'my_pretrained_models/reports/undersampling/classification_report{prefix}_e{epoch}.csv')


def test_enclosure():
    x = []

    def closure(hoge: str = None):
        if hoge:
            nonlocal x
            x.append(hoge)
        if hoge is None:
            return x
    return closure


if __name__ == "__main__":
    gen_report()
    # # model.vecをload
    # model = gensim.models.KeyedVectors.load_word2vec_format(
    #     'hoge.vec', binary=False)

    # # バイナリファイルとして保存
    # model.save_word2vec_format(
    #     "hoge_fixed.bin", binary=True)
    # print(model.most_similar('イヤホン'))
    pass
