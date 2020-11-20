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


if __name__ == "__main__":
    pass
