from operator import index
import pathlib
import numpy as np
from pathlib import Path
import gensim
import livedoor_news_corpus
import dev_func
import mecab_de_wakatiko
import pandas as pd


class SADGen():
    def __init__(self, src: Path) -> None:
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            src, binary=True)
        self.positive_words = None
        self.negative_words = None
        self.positive_evals = None
        self.negative_evals = None

    def set_posi_nega_list(self, positive_words: list = None, positive_evals: list = None,  negative_words: list = None, negative_evals: list = None):
        """極性辞書の極性値の基準となる単語リストをセットします。

        Parameters
        -----
        positive_words (list, optional): 一般的に「極めてポジティブ」な意味の単語群。
        positive_evals (list, optional): 一般的に「極めてネガティブ」な意味の単語群。
        negative_words (list, optional): 任意の「極めてポジティブ」な意味の単語群。
        negative_evals (list, optional): 任意の「極めてネガティブ」な意味の単語群。
        """
        # 一般的に「極めてポジティブ」な意味の単語群
        if positive_words is None:
            self.positive_words = ['優れる', '良い', '喜ぶ', '褒める', 'めでたい', '賢い', '善い', '適す', '天晴', '麗しい'
                                   '祝う', '功績', '嬉しい', '喜び', '才知', '徳', '才能', '素晴らしい', '芳しい', '称える',
                                   '適切', '崇める', '助ける', '抜きんでる', '清水', '雄雄しい', '仕合せ', '幸い', '吉兆', '秀でる']
        else:
            self.positive_words = positive_words

        # 一般的に「極めてネガティブ」な意味の単語群
        if negative_words is None:
            self.negative_words = ['悪い', '死ぬ', '病気', '酷い', '罵る', '卑しい',
                                   '下手', '苦しむ', '苦しい', '付く', '厳しい', '難しい', '殺す', '難い', '荒荒しい',
                                   '惨い', '責める', '敵', '背く', '嘲る', '苦しめる', '辛い', '物寂しい', '罰', '不貞腐る',
                                   '寒い', '下らない', '残念']
        else:
            self.negative_words = negative_words

        # 任意の「極めてポジティブ」な意味の単語群
        self.positive_evals = positive_evals
        # 任意の「極めてネガティブ」な意味の単語群
        self.negative_evals = negative_evals

    # @dev_func.measure_processing_speed
    def calc_similarity_mean(self, ruler: list, keyword: str):
        '''
        単語群とキーワードとの類似度の平均を計算します。
        '''
        loop = 0
        similarity_value = 0

        for w in ruler:
            try:
                s = self.model.similarity(w, keyword)
                similarity_value += s
                loop += 1
            except:
                continue
        try:
            similarity_mean = similarity_value / loop
        except:
            similarity_mean = 0

        return similarity_mean

    # @dev_func.measure_processing_speed
    def calc_similarity_mean2(self, ruler: list, keyword: str):
        '''
        単語群とキーワードとの類似度の平均を計算します。
        '''
        similarity_values = []
        for w in ruler:
            try:
                s = self.model.similarity(w, keyword)
                similarity_values.append(s)
            except:
                continue
        try:
            similarity_mean = sum(similarity_values)/len(similarity_values)
        except:
            similarity_mean = 0

        return similarity_mean

    def posi_nega_score(self, keyword):
        # 一般的に極めてポジティブな意味の単語群との類似度
        mean_posi_words = self.calc_similarity_mean(
            self.positive_words, keyword)
        # 一般的に極めてネガティブな意味の単語群との類似度
        mean_nega_words = self.calc_similarity_mean(
            self.negative_words, keyword)

        # 一般的に極めてポジティブかネガティブへ類似度の高い数値を一時的に格納
        if mean_nega_words < mean_posi_words:
            tmp_words = mean_posi_words
        elif mean_posi_words < mean_nega_words:
            tmp_words = -mean_nega_words
        else:
            tmp_words = 0

        # 任意の単語群が両者とも設定されていない場合は一般的にポジティブかネガティブかの類似度を返す。
        # 一方のみの単語群の設定は後の計算でTypeErrorが発生。
        if self.positive_evals is self.negative_evals is None:
            return tmp_words

        # 任意の極めてポジティブな意味の単語群との類似度
        mean_posi_evals = self.calc_similarity_mean(
            self.positive_evals, keyword)
        # 任意の極めてネガティブな意味の単語群との類似度
        mean_nega_evals = self.calc_similarity_mean(
            self.negative_evals, keyword)

        # 任意の極めてポジティブかネガティブへ類似度の高い数値を一時的に格納
        if mean_nega_evals < mean_posi_evals:
            tmp_eval = mean_posi_evals
        elif mean_posi_evals < mean_nega_evals:
            tmp_eval = -mean_nega_evals
        else:
            tmp_eval = 0

        print(mean_posi_words, mean_posi_evals,
              mean_nega_words, mean_nega_evals)
        # 類似度の絶対値の大きさは一般的あるいは任意の単語群との類似度の大きさを示す。
        if abs(tmp_eval) < abs(tmp_words):
            return tmp_words
        elif abs(tmp_words) < abs(tmp_eval):
            return tmp_eval
        else:
            return 0

    def add_sentiment_score(self, df):
        # 各単語にスコアを割り振る
        df['スコア'] = df['単語'].apply(lambda x: self.posi_nega_score(x))
        # 与えられたスコアを-1から1の範囲に調整
        score = np.array(df['スコア'])
        score_std = (score - score.min())/(score.max() - score.min())
        score_scaled = score_std * (1 - (-1)) + (-1)
        df['スコア'] = score_scaled
        return df


def check_polarity():
    src = 'output/20201101/wiki_model_20201101_fixed.bin'
    negative_eval = ['ノイズ', '途切れる', '邪魔', '遅延',
                     '音漏れ', '雑音', '重い', '中華', '故障', '初期不良', '痛い']
    positive_eval = ['軽い', 'コンパクト', '高音', '低音', '綺麗'
                     'フィット', '無線', 'クリア', '長時間', 'スムーズ', '重低音', '遮音']
    osenti = SADGen(Path(src))
    osenti.set_posi_nega_list(
        negative_evals=negative_eval, positive_evals=positive_eval)
    # return osenti
    while True:
        print('極性値を調べたい単語を入力してください')
        word = input()
        if word == 'q':
            break
        score = osenti.posi_nega_score(word)
        # with open(pathlib.Path('positive_evals.txt'), mode='a') as f:
        #     print('単語を登録しますか？')
        #     if input() == 'y':
        #         f.write()
        # words = osenti.model.most_similar(positive=['ノイズ'], topn=10)
        print(score)


def gen_sad():

    src = 'output/20201101/wiki_model_20201101_fixed.bin'
    negative_eval = ['ノイズ', '途切れる', '邪魔', '遅延',
                     '音漏れ', '雑音', '重い', '中華', '故障', '初期不良', '痛い']
    positive_eval = ['軽い', 'コンパクト', '高音', '低音', '綺麗'
                     'フィット', '無線', 'クリア', '長時間', 'スムーズ', '重低音', '遮音']
    osenti = SADGen(Path(src))
    osenti.set_posi_nega_list(
        negative_evals=negative_eval, positive_evals=positive_eval)
    osenti.set_posi_nega_list()
    wakatiko = mecab_de_wakatiko.wakatiko()
    df = wakatiko.extract_words_mecab_lemma()
    df = osenti.add_sentiment_score(df)
    df.to_csv(pathlib.Path('resources/sad/earphone.sad'), index=False)


if __name__ == "__main__":
    src = 'output/20201101/wiki_model_20201101_fixed.bin'
    # negative_eval = ['ノイズ', '途切れる', '邪魔', '遅延',
    #                  '音漏れ', '雑音', '重い', '中華', '故障', '初期不良', '痛い']
    # positive_eval = ['軽い', 'コンパクト', '高音', '低音', '綺麗'
    #                  'フィット', '無線', 'クリア', '長時間', 'スムーズ', '重低音', '遮音']
    osenti = SADGen(Path(src))
    result = osenti.model.most_similar('ダイエット')
    print(result)
    # osenti.set_posi_nega_list(
    #     negative_evals=negative_eval, positive_evals=positive_eval)
    # osenti.set_posi_nega_list()
    # gen_sad()
    pass
