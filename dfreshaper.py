import pandas as pd
from pathlib import Path
from my_module import compressor
from my_module import text_formatter as tf
import re


def aggregate_by_rate(review_path: Path, freq: str = 'M') -> pd.DataFrame:
    '''
    任意の期間ごとに各評価のレビュー数を集計します
    '''
    # Read DataFrame.
    df = pd.read_csv(review_path)
    # Delete 'comments' colum.
    df.drop(columns=['comments', 'votes'], inplace=True)
    # Convert 'dates' colum into DateTime type
    df['dates'] = pd.to_datetime(df['dates'])
    # Set index to 'dates' coulum.
    df.set_index('dates', inplace=True)
    # Aggregate review by any period.
    df = df.groupby(pd.Grouper(freq=freq))
    # Generate template.
    df_result = pd.DataFrame(columns=range(1, 6))
    for period, group in df:
        # 評価ごとにレビュー数を集計.インデックスを昇順にソート.行と列の入れ替え
        group = group.apply(pd.value_counts).sort_index().T
        # Rename Index to Period.
        group = group.rename(index={'rates': period})
        # Merge DataFrame.
        df_result = pd.concat([df_result, group])

    # Replace Nan in DataFrame with 0.
    df_result.fillna(0, inplace=True)
    # Formats DateTimeIndex to '%y-%m-%d'and converts it to String type.
    df_result.index = df_result.index.strftime('%y-%m-%d')
    # Insert total colum.
    df_result[15] = df_result.sum(axis=1).astype('int64')
    # Reduce memory usage for pandas DataFrame.
    df_result = compressor.reduce_mem_usage(df_result)

    # Return
    return df_result

    # 任意の分割数で分割
    # print((np.array_split(df['rates'].values, 3)))


def fmt_reviews(review_path: Path):
    '''
    レビューCSVから日付、レート、コメントを抽出してCSVへ出力します。
    '''
    # フォーマットしたファイルのパスをセット
    formatted_file_path = review_path.with_name(f'f{review_path.name}')
    # CSVを分割読込み
    df = pd.read_csv(review_path)
    # 処理に使用する列を抽出
    df = df[['dates', 'comments', 'rates', 'votes']]
    # 英文のレビューを削除
    pattern = r'[a-zA-Z0-9\W_]*$'
    repatter = re.compile(pattern)
    drop_index = df.index[df['comments'].str.match(repatter)]
    df.drop(drop_index, inplace=True)
    # コメントを整形
    df['comments'] = df['comments'].apply(tf.fmt_comments)
    # 日付をdatetime形式へ変換
    df['dates'] = df['dates'].apply(tf.convert_datetime_date)
    # 評価の数値を抽出
    df['rates'] = df['rates'].apply(tf.extract_rate)
    # 投票数を抽出
    df.fillna({'votes': 0}, inplace=True)
    df['votes'] = df['votes'].apply(tf.extract_vote).astype('int8')
    # 日付をインデックスへセット
    df.set_index('dates', inplace=True)
    df.to_csv(formatted_file_path)


def fmt_labeled_review(review_path: Path):
    # ラベルを付与したCSVの出力先
    labeled_file_path = review_path.with_name(
        f'{review_path.stem}_labeled.csv')
    # CSVの読み込み
    df = pd.read_csv(review_path)
    # 必要な情報以外を削除
    df = df[['comments', 'rates']]
    # ラベルを付与
    df['label'] = df['rates'].apply(lambda x: f'__label__{x}')
    df = df[['label', 'comments']]
    # CSVへ出力
    df.to_csv(labeled_file_path, index=False)


def split_review_by_sent(path: Path):
    df = pd.read_csv(path)
    rates = range(1, 5)
    for i in rates:
        df_temp = df[df['rates'] == i]
        pattern = r'[。.?!]'
        repatter = re.compile(pattern)
        df_temp['comments'] = df_temp['comments'].apply(
            lambda x: [sent for sent in re.split(repatter, str(x))])
        comments = [
            comment for sublist in df_temp['comments'].values for comment in sublist if len(comment) > 1]
        df_temp = pd.DataFrame(comments, columns=['comments'])
        df_temp.to_csv(path.with_name(f'{path.stem}_rate{i}.csv'))


if __name__ == "__main__":
    pass
