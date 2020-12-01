import pandas as pd
from my_module import path_manager as pm
from distutils import dir_util
import re
from plotter import ReviewPlotter
import dfreshaper
import pathlib
import gcp_sentiment
import time
import json

# インデックスをプライマリーキーにASINを外部参照
# ASINを元にテーブルを作成


def get_dirs(path: str):
    dirs = [p for p in pathlib.Path(path).iterdir() if p.is_dir()]
    return dirs


def get_dirs_name(path: str):
    dirs = get_dirs(path)
    dirs = list(map(lambda x: x.stem, dirs))
    return dirs


def marge_csv(path: str, file_name: str):
    path = pathlib.Path(path)
    paths = path.glob('*.csv')
    # csvファイルの中身を追加していくリストを用意
    data_list = [pd.read_csv(path) for path in paths]
    # リストを全て縦に結合
    df = pd.concat(data_list, axis=0, sort=True)
    df.to_csv(file_name, index=False)


def get_all_reviews_paths(is_format: bool = False):
    '''
    Return the generator for all amazon review paths.
    '''
    # レビューファイルのパス
    reviews_path = pm.get_abs_path(__file__, 'csv/reviews/')
    # Windowsはパスをエスケープ文字'\'とバックスラッシュ'\'の'\\'でパスを表現している。
    pattern = r'[/|\\]+(?![a-z])[A-Z0-9]+\.csv'
    repatter = re.compile(pattern)
    if is_format is False:
        # 辞書内包表記 {key:"拡張子のないファイル名",item:"ファイルのパス"}
        paths_gen = [p for p in reviews_path.glob(
            '**/*.csv') if re.search(repatter, str(p))]
    else:
        # 辞書内包表記 {key:"拡張子のないファイル名",item:"ファイルのパス"}
        paths_gen = [p.with_name(f'f{p.name}') for p in reviews_path.glob(
            '**/*.csv') if re.search(repatter, str(p))]

    return paths_gen


def copy_all_reviews():
    '''
    コピー元ファイルの更新日時が古い場合はコピー先のファイルを上書きをしない。
    非推奨：コマンド打ったほうが簡単
    '''
    # コピー元のレビューディレクトリ
    # reviews_path_src = r'C:/Users/yaroy/Documents/Python_Projects/Crawler/Amazon/csv/reviews/'
    reviews_path_src = r'/home/ubuntu/Documents/python_projects/py37/Web_Crawler/Amazon/csv/reviews/'
    reviews_path_src = pathlib.Path(reviews_path_src)
    reviews_path_src = str(reviews_path_src)
    # コピー先のレビューディレクトリ
    reviews_path_dst = pm.get_abs_path(__file__, 'csv/reviews/')
    reviews_path_dst = str(reviews_path_dst)
    # 更新/差分ファイルをコピー
    dir_util.copy_tree(reviews_path_src, reviews_path_dst, update=1)


def plot_chronological(ratio: bool):
    paths = get_all_reviews_paths(is_format=True)
    for path in paths:
        df = dfreshaper.aggregate_by_rate(path)
        r_plotter = ReviewPlotter(headless=True)
        # r_plotter.plot_stacked_bar_graph(df, ratio=ratio)
        r_plotter.plot_line_of_rate(df, ratio=ratio)
        if ratio:
            path = pm.get_abs_path(
                path, f'plots/{path.stem[1:]}_line_ratio.png')
            # path, f'plots/{path.stem[1:]}_chronological_ratio.png')
        else:
            path = pm.get_abs_path(
                path, f'plots/{path.stem[1:]}_line.png')
            # path, f'plots/{path.stem[1:]}_chronological.png')

        r_plotter.save(path)

    # r_plotter.plot_histgram_of_rate(df)
    # r_plotter.plot()


def test_plot2():
    path = 'C:/Users/yaroy/Documents/Python_Projects/NLP/csv/reviews/B07VR7ZBBQ/fB07VR7ZBBQ.csv'
    path = 'C:/Users/yaroy/Documents/Python_Projects/NLP/csv/reviews/B019GNUT0C/fB019GNUT0C.csv'
    path = pathlib.Path(path)
    df = dfreshaper.aggregate_by_rate(path)
    r_plotter = ReviewPlotter()
    # r_plotter.plot_line_of_rate(df, ratio=True)
    # r_plotter.plot_stacked_bar_graph(df, ratio=True)
    r_plotter.plot_stacked_bar_graph(df)
    r_plotter.plot()


def csv2txt(path: pathlib.Path):
    df = pd.read_csv(path)
    df = df[df['rates'] < 5]
    comments = df['comments'].values
    txt_path = path.with_name(f'{path.stem}_commnets.txt')
    with open(txt_path, mode='w', encoding='utf-8') as f:
        # f.write('\n'.join(comments))
        f.writelines(comments)


def count_character():
    chars = 0

    def count(path: pathlib.Path):
        txt_path = path.with_name(f'{path.stem}_commnets.txt')
        with open(txt_path, mode='r') as f:
            sent = f.readline()
            nonlocal chars
            chars = chars + len(sent)
            return chars
    return count


def fetch_sentiment(path: pathlib.Path):
    time.sleep(1)
    txt_path = path.with_name(f'{path.stem}_commnets.txt')
    json_path = path.with_suffix('.json')
    with open(txt_path, mode='r') as f:
        text = f.read()
        text = gcp_sentiment.analyze_sentiment_by_requests(text)
    with open(json_path, mode='w', encoding='utf-8') as f:
        f.write(text)
    print(f'output {json_path}')


def marge_senti_json():

    def json2df(path: pathlib.Path):
        with open(path.with_name(f'{path.stem}.json'), mode='r') as f:
            df = pd.json_normalize(json.load(f)['sentences'], sep='_')
            df.rename(columns=lambda x: x.split('_')[1], inplace=True)
            return df

    paths = get_all_reviews_paths(is_format=True)
    df_list = [json2df(path) for path in paths]
    df = pd.concat(df_list, axis=0, sort=True)
    df.to_csv("sent_senti.csv", index=False)


if __name__ == "__main__":
    # path = pathlib.Path('earphone_review.csv')
    # dfreshaper.fmt_reviews(path)
    # path = pathlib.Path('fearphone_review.csv')
    # split_review_by_sent(path)
    # test_plot2()
    # plot_chronological(True)
    # paths = get_all_reviews_paths(is_format=True)
    # for path in paths:
    #     json_path = path.with_name(f'{path.stem}.json')
    #     with open(json_path, mode='r') as f:
    #         json_tmp = json.load(f)
    #         df = pd.io.json.json_normalize(json_tmp['sentences'])

    # csv2txt(path)
    # dfreshaper.fmt_reviews(path)
    # marge_csv()
    # copy_all_reviews()
    pass
