from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from my_module import path_manager as pm


def txt2d2vmodel(ASIN: str):
    '''
    テキストからDoc2Vecモデルを生成
    '''
    # 絶対パスをセット
    path = f'csv/reviews/{ASIN}/wakati/wakati.txt'
    path = pm.get_abs_path(__file__, path)
    with open(path, mode='r', encoding='utf_8') as f:
        # １文書ずつ、単語に分割してリストへ格納[([単語1,単語2,単語3],文書id),...]
        # words：文書に含まれる単語のリスト（単語の重複あり）
        # tags：文書の識別子（リストで指定．1つの文書に複数のタグを付与できる）
        trainings = [TaggedDocument(words=data.split(), tags=[i])
                     for i, data in enumerate(f)]

        # トレーニング
        m = Doc2Vec(documents=trainings, dm=1, vectorize=300,
                    window=8, min_count=10, workers=4)
        # モデルのセーブ
        m.save("./doc2vec.model")
