import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import oseti


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


if __name__ == "__main__":
    rename_df_col()
    pass
