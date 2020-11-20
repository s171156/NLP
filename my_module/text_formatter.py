import re
import datetime as dt
import neologdn


def remove_url(text):
    '''
    正規表現で取得したテキストのURLを''に置換します。
    '''
    # 正規表現で使用するパターンをコンパイル
    pattern = r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+"
    repatter = re.compile(pattern)
    return re.sub(repatter, '', text)


def format_tweets(text):
    '''
    MeCabに入れる前のツイートの整形
    '''
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    text = re.sub('RT', "", text)
    text = re.sub(r'[!-~]', "", text)  # 半角記号,数字,英字
    text = re.sub(r'[︰-＠]', "", text)  # 全角記号
    text = re.sub('\n', " ", text)  # 改行文字
    return text


def fmt_comments(text: str):
    '''
    Ginza解析前の文章の整形
    '''
    # URLを除去
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    # mecabの形態素解析では25文字以上の英字は分割されるため、26文字以上の英字は除去。
    text = re.sub(r'[a-zA-Z0-9\W_]{26,}', '', text)
    # 繰り返し表現や半角全角を吸収。
    text = neologdn.normalize(text)
    # 空白文字を除去
    text = re.sub(r'\s', "", text)
    # 数字を0に置換
    # text = re.sub(r'\d', '0', text)
    # 半角英数字以外、半角記号を除去
    # text = re.sub(r'[\W]', "", text)
    # 文末記号以外を削除
    text = re.sub(r'[^。、!\?\w]', '', text)
    # 文末記号がない場合は記号を付与。
    if text[-1] not in ['。', '!', '?']:
        text = text + '。'
    # テキストのアルファベットを小文字で統一
    text = text.lower()
    return text


def extract_date_ymd(text: str) -> list:
    '''
    文字列から「YYYY年MM月DD日」を抽出します。
    '''
    pattern = r'\d{1,4}年\d{1,2}月\d{1,2}日'
    repatter = re.compile(pattern)
    return re.search(repatter, text).group()


def disassemble_date_ymd(date: str) -> list:
    '''
    「YYYY年MM月DD日」文字列から年月日を抽出します。
    '''
    pattern = r'\d{1,4}年|\d{1,2}月|\d{1,2}日'
    repatter = re.compile(pattern)
    date = re.findall(repatter, date)
    date = [d[:-1] for d in date]
    date = list(map(int, date))
    return date


def convert_datetime_date(date: str):
    '''
    DateTime_Dateオブジェクトに変換します。
    '''
    date = extract_date_ymd(date)
    date = disassemble_date_ymd(date)
    return dt.date(*date)


def extract_rate(text):
    '''
    Amazonレビューから評価を抽出します。
    '''
    pattern = r'\d\.0'
    repatter = re.compile(pattern)
    text = re.search(repatter, text).group()
    return int(text[0])


def extract_vote(text):
    # 引数の型が'str'
    if type(text) is str:
        return int(text[0])
    return 0


if __name__ == "__main__":
    pass
