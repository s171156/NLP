import MeCab
import pandas as pd
import collections
import pathlib
from typing import Union, List


class wakatiko:
    def __init__(self) -> None:
        self.tagger = MeCab.Tagger(
            '-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')

    def make_unique_words(self, sentences: Union[List[str], str], is_lemma: bool = False, lexical_category: List[str] = ['名詞', '形容詞', '動詞']) -> List[str]:
        """一意の単語リストを生成します。

        Args:
            sentences (Union[List[str], str]): 重複のある文字列のリスト
            is_lemma (bool, optional): 基本形. Defaults to False.
            lexical_category (List[str], optional): 品詞の指定. Defaults to ['名詞', '形容詞', '動詞'].

        Returns:
            List[str]: 一意の単語リスト
        """

        # sentencesの引数がstrの場合はlistに変換
        if type(sentences) is str:
            sentences = list(sentences)

        # sentencesから改行を除去
        sentences = map(lambda x: str(x).rstrip(), sentences)

        if is_lemma:
            # 単語の基本形をリストに格納
            words = [chunk.split('\t')[1].split(',')[6] for sentence in sentences for chunk in self.tagger.parse(
                sentence).splitlines()[:-1] if chunk.split('\t')[1].split(',')[0] in lexical_category]
        else:
            # 分かち書きされた単語をリストに格納
            words = [chunk.split('\t')[0] for sentence in sentences for chunk in self.tagger.parse(
                sentence).splitlines()[:-1] if chunk.split('\t')[1].split(',')[0] in lexical_category]
        # 単語の重複を除去
        words = list(set(words))
        return words

    def measure_word_length(self, sentences: List[str], lexical_category: List[str] = ['名詞', '形容詞', '動詞']):
        """リストに含まれる文字列の単語の長さを表示します。

        Args:
            sentences (List[str]): 単語の長さを調べたい文字列のリスト
            lexical_category (List[str], optional): 品詞の指定. Defaults to ['名詞', '形容詞', '動詞'].
        """
        sentences = map(lambda x: str(x).rstrip(), sentences)
        # 名詞、形容詞、動詞の基本形をリストに格納
        words = [chunk.split('\t')[0] for sentence in sentences for chunk in self.tagger.parse(
            sentence).splitlines()[:-1] if chunk.split('\t')[1].split(',')[0] in lexical_category]
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

    def count_word_freq(self, sentences: List[str], lexical_category: List[str] = ['名詞', '形容詞', '動詞']) -> pd.DataFrame:
        """文字列リスト中の単語の出現頻度をデータフレームにします。

        Args:
            sentences (List[str]): 単語の出現頻度を調査したい文字列リスト
            lexical_category (List[str], optional): 品詞の指定. Defaults to ['名詞', '形容詞', '動詞'].

        Returns:
            pd.DataFrame: 単語の出現頻度のデータフレーム
        """

        # データフレームの読み込み
        df = pd.read_csv(pathlib.Path('resources/review/fearphone_review.csv'))
        # 結果を格納する変数
        result_df = None
        for i in range(1, 6):
            tmp_df = df[df['rates'] == i]
            sentences = tmp_df['comments'].apply(
                lambda x: str(x).rstrip()).values
            words = [chunk.split('\t')[1].split(',')[6] for sentence in sentences for chunk in self.tagger.parse(
                sentence).splitlines()[:-1] if chunk.split('\t')[1].split(',')[0] in lexical_category]
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

        return result_df

    def count_word_freq2(self, sentences: List[str], lexical_category: List[str] = ['名詞', '形容詞', '動詞']) -> pd.DataFrame:
        """文字列リスト中の単語の出現頻度をデータフレームにします。

        Args:
            sentences (List[str]): 単語の出現頻度を調査したい文字列リスト
            lexical_category (List[str], optional): 品詞の指定. Defaults to ['名詞', '形容詞', '動詞'].

        Returns:
            pd.DataFrame: 単語の出現頻度のデータフレーム
        """
        words = [chunk.split('\t')[1].split(',')[6] for sentence in sentences for chunk in self.tagger.parse(
            sentence).splitlines()[:-1] if chunk.split('\t')[1].split(',')[0] in lexical_category]
        # 単語の出現頻度をカウント
        words = collections.Counter(words)
        return words

    def wakati_sentence(self, sentence: str, is_lemma: bool = False) -> str:
        """文字列を分かち書きします。

        Args:
            sentence (str): 分かち書きしたい文字列
            is_lemma (bool, optional): 基本形の指定. Defaults to False.

        Returns:
            str: 分かち書き（単語ごとに空白で区切ること）した文字列
        """
        if is_lemma:
            sentence = ' '.join([chunk.split('\t')[1].split(
                ',')[6] for chunk in self.tagger.parse(str(sentence)).splitlines()[:-1]])
        else:
            sentence = ' '.join([chunk.split('\t')[0] for chunk in self.tagger.parse(
                str(sentence)).splitlines()[:-1]])

        # sentence = ' '.join([chunk.split('\t')[1].split(',')[6] for chunk in self.tagger.parse(
        #     str(sentence)).splitlines()[:-1] if chunk.split('\t')[1].split(',')[0] in ['動詞', '名詞', '形容詞', '助動詞']])

        return sentence


if __name__ == "__main__":
    wakati = wakatiko()
    # path = pathlib.Path('sent_senti_labeled.csv')
    path = pathlib.Path('resources/review/fearphone_review_labeled.csv')
    # path = pathlib.Path('resources/review/fearphone_review.csv')
    # データフレームの読み込み
    df = pd.read_csv(path)
    # df['content'] = df['content'].apply(wakati.wakati_sentence)
    df['comments'] = df['comments'].apply(wakati.wakati_sentence)
    df.to_csv(f'{path.stem}_wakati.csv', index=False)
    # sentences = df['comments'].values[0]
    # sentence = wakati.wakati_sentence(sentences, False)
    # print(sentence)
    pass
