import MeCab
import pandas as pd
import collections
import pathlib


class wakatiko:
    def __init__(self) -> None:
        self.tagger = MeCab.Tagger(
            '-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')

    def set_tagger_options(self, *args):
        options = ' '.join(list(args))
        self.tagger = MeCab.Tagger(
            f'{options} -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')

    def extract_words(self):
        article_df = pd.read_csv('ldcc-20140209/csv/lnp.csv')
        # スポーツ関連の記事に絞る
        news_df = article_df.query('Name == "sports-watch"')
        # news_df = article_df[article_df['Name'] == 'sports-watch']
        # 記事の内容を分かち書きしてリスト化
        news_df['Body'] = news_df['Body'].apply(
            lambda x: self.tagger.parse(x).rstrip().split(' '))
        # ネストしたリストを展開して単語を抽出
        words = [word for sublist in news_df['Body'].values for word in sublist]
        # 単語の重複を除去
        words = list(set(words))
        # データフレームの生成
        word_df = pd.DataFrame(words, columns=['単語'])
        return word_df

    def extract_words_mecab_lemma(self):
        # データフレームの読み込み
        df = pd.read_csv(pathlib.Path('resources/review/fearphone_review.csv'))
        # 文章を形態素解析
        sentences = df['comments'].apply(lambda x: str(x).rstrip()).values
        # 名詞、形容詞、動詞の基本形をリストに格納
        words = [chunk.split('\t')[1].split(',')[6] for sentence in sentences for chunk in self.tagger.parse(
            sentence).splitlines()[:-1] if chunk.split('\t')[1].split(',')[0] in ['名詞', '形容詞', '動詞']]
        # 単語の重複を除去
        words = list(set(words))
        # データフレームの生成
        word_df = pd.DataFrame(words, columns=['単語'])
        return word_df

    def measure_word_length(self):
        # データフレームの読み込み
        df = pd.read_csv(pathlib.Path(
            'resources/review/fearphone_review copy.csv'))
        # 文章を形態素解析
        sentences = df['comments'].apply(lambda x: str(x).rstrip()).values
        # 名詞、形容詞、動詞の基本形をリストに格納
        words = [chunk.split('\t')[0] for sentence in sentences for chunk in self.tagger.parse(
            sentence).splitlines()[:-1] if chunk.split('\t')[1].split(',')[0] in ['名詞', '形容詞', '動詞']]
        # 単語の重複を除去
        words = list(set(words))
        wlength = list(map(lambda x: len(x), words))
        wlength_set = list(set(wlength))
        wlen_mean = sum(wlength_set) / len(wlength_set)
        wlen_min = min(wlength_set)
        wlen_max = max(wlength_set)
        print(f'単語の平均文字数: {wlen_mean}')
        print(f'単語の最小文字数: {wlen_min}')
        print(f'最短の単語: {words[wlength.index(wlen_min)]}')
        print(f'単語の最大文字数: {wlen_max}')
        print(f'最長の単語: {words[wlength.index(wlen_max)]}')

    def count_word_freq(self):
        # データフレームの読み込み
        df = pd.read_csv(pathlib.Path('resources/review/fearphone_review.csv'))
        # 結果を格納する変数
        result_df = None
        for i in range(1, 6):
            tmp_df = df[df['rates'] == i]
            sentences = tmp_df['comments'].apply(
                lambda x: str(x).rstrip()).values
            words = [chunk.split('\t')[1].split(',')[6] for sentence in sentences for chunk in self.tagger.parse(
                sentence).splitlines()[:-1] if chunk.split('\t')[1].split(',')[0] in ['名詞', '形容詞', '動詞']]
            # 単語の出現頻度をカウント
            words = collections.Counter(words)
            rate_list = [i] * len(words)
            # 単語の出現頻度を辞書に格納する
            word_dict = {'単語': list(words.keys()), '頻度': list(
                words.values()), '評価': rate_list}
            # 単語の出現頻度の辞書からデータフレームを作成
            word_df = pd.DataFrame.from_dict(word_dict)
            word_df.sort_values('頻度', ascending=False, inplace=True)
            # データフレームを代入あるいは結合
            if result_df is None:
                result_df = word_df
            else:
                result_df = pd.concat([result_df, word_df], axis=0)
        # データフレームをCSVに出力
        result_df.to_csv(pathlib.Path('word_freq.csv'), index=False)

    def review_wakati_lemma(self):
        df_path = pathlib.Path('resources/review/fearphone_review_labeled.csv')
        df = pd.read_csv(df_path)

        df['comments'] = df['comments'].apply(
            lambda x: ' '.join([chunk.split('\t')[1].split(',')[6] for chunk in self.tagger.parse(str(x)).splitlines()[:-1]]))

        # return df
        wakati_path = df_path.with_name(f'{df_path.stem}_wakati.csv')
        df.to_csv(wakati_path, index=False)

    def review_wakati_owakati(self):
        self.set_tagger_options('-Owakati')
        df_path = pathlib.Path(
            'resources/review/fearphone_review_labeled.csv')
        df = pd.read_csv(df_path)

        df['comments'] = df['comments'].apply(
            lambda x: self.tagger.parse(str(x))[:-1])

        # return df
        wakati_path = df_path.with_name(f'{df_path.stem}_wakati.csv')
        df.to_csv(wakati_path, index=False, header=False, sep=' ')


if __name__ == "__main__":
    wakati = wakatiko()
    # wakati.review_wakati_owakati()
    wakati.measure_word_length()
    # path = pathlib.Path('resources/review/fearphone_review.csv')
    # freq = path.with_name('word_freq2.csv')
    # df = pd.read_csv(freq)
    # tmp = df[df['評価'] == 5][:30]['単語']
    # print(tmp)

    # tmp = df[df['評価'] == 1][:30]['単語']
    # print(tmp)
    pass
