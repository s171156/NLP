import pathlib
import re
import pandas as pd


def lnc_txt2csv():
    p_temp = pathlib.Path('ldcc-20140209/text')

    article_list = []

    # フォルダ内のテキストファイルを全てサーチ
    for p in p_temp.glob('**/*.txt'):
        # フォルダ名からニュースサイトの名前を取得
        media = str(p.parent.stem)
        # 拡張子を除くファイル名を取得
        file_name = str(p.stem)

        if file_name != 'LICENSE.txt':
            # テキストファイルを読み込む
            with open(p, 'r') as f:
                # テキストファイルの中身を一行ずつ読み込み、リスト形式で格納
                article = f.readlines()
                # 不要な改行等を置換処理
                article = [re.sub(r'[\s\u3000]', '', i) for i in article]
            # ニュースサイト名・記事URL・日付・記事タイトル・本文の並びでリスト化
            article_list.append([media, article[0], article[1],
                                 article[2], ''.join(article[3:])])
        else:
            continue

    article_df = pd.DataFrame(article_list, columns=[
                              'Name', 'URL', 'Date', 'Title', 'Body'])
    path_lnp_csv = p_temp.parent.joinpath('csv/lnp.csv')
    path_lnp_csv.parent.mkdir(parents=True, exist_ok=True)
    article_df.to_csv(path_lnp_csv)


def extract_words_in_lnc(self, category: str):
    '''
    livedoor_news_corpusから単語を抽出します。
    '''
    article_df = pd.read_csv('ldcc-20140209/csv/lnp.csv')
    # スポーツ関連の記事に絞る
    news_df = article_df[article_df['Name'] == category]
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


if __name__ == "__main__":
    pass
