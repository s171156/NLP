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


if __name__ == "__main__":
    text = 'hello world'
    print(text[-1] not in ['!', "?", '。'])
    print(text[-1] in ['!', "?", '。'] is False)
    if text[-1] in ['!', "?", '。'] is False:
        print(True)
        text = text + '!'
    print(text)
    pass
