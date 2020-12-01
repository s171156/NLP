from ginza.bunsetu_recognizer import bunsetu_phrase_span, bunsetu_span
from my_module import text_formatter as tf
from collections import Counter
from spacy import displacy
import numpy as np
from typing import Tuple, List
import spacy
from pathlib import Path
import pandas as pd
from my_module import path_manager as pm
from ginza import bunsetu_recognizer as br
import ginza
# Ginzaの辞書をロード
nlp = spacy.load('ja_ginza')


def similarity_matrix(texts: List[str]) -> List[List[np.float64]]:
    """
    テキスト同士の類似度を計算する。
    Parameters
    ----------
    texts : str
        日本語文のリスト。
    Returns
    -------
    List[List[np.float64]]
        文同士の類似度。
    Notes
    -----
    spaCy の Doc.similarity (https://spacy.io/api/doc#similarity) を使っている。
    """
    docs = [nlp(text) for text in texts]
    rows = [[doc.similarity(comp) for doc in docs] for comp in docs]
    df = pd.DataFrame(rows)
    return df


def ginza_de_wakatiko(texts: list, is_nva: bool = False, is_lemma: bool = False):
    '''
    テキストを分かち書きします。
    '''
    # パイプ処理
    docs = list(nlp.pipe(texts, disable=['ner']))

    # 可読性重視
    # sentences = []
    # for doc in docs:
    #     tokens = [token.text for token in doc]
    #     sentences.append(' '.join(tokens))

    if is_nva:
        # 品詞のうち名詞/動詞/形容詞を抽出
        sentences = (' '.join([token.text for token in doc if token.pos_ in [
            'NOUN', 'VERB', 'ADJ']]) for doc in docs)
    elif is_lemma:
        # 基本形を抽出
        sentences = (' '.join([token.lemma_ for token in doc]) for doc in docs)
    elif is_nva and is_lemma:
        # 品詞のうち名詞/動詞/形容詞を基本形で抽出
        sentences = (' '.join([token.lemma_ for token in doc if token.pos_ in [
            'NOUN', 'VERB', 'ADJ']]) for doc in docs)
    else:
        sentences = (' '.join([token.text for token in doc]) for doc in docs)

    return sentences


def output_comments_txt(review_path: Path, freq: str = 'M', is_div: bool = False):
    '''
    レビューのコメントを月別にtxt形式で出力します。
    '''
    # データフレームの読込み
    df = pd.read_csv(review_path)
    # 日付のカラムをdatetime形式へ変換
    df['dates'] = pd.to_datetime(df['dates'])
    # 日付のカラムをインデックスへセット
    df.set_index('dates', inplace=True)
    # レビューを月別に集計
    grouped = df.groupby(pd.Grouper(freq=freq))

    if is_div:
        # 月別に集計したタプルから日付とコメントを取り出し
        for idx, group in grouped:
            # 分かち書きしたコメントをリストへ格納
            # text_wakati = ginza_de_wakatiko(group['comments'].values)
            text_wakati = ginza_de_wakatiko(
                group['comments'].values, True, True)
            idx = idx.strftime('%Y-%m-%d')
            # 分かち書きしたテキストを出力するファイルのパスをセット
            # wakati_path = pm.get_abs_path(review_path, f'wakati/{idx}.txt')
            wakati_path = pm.get_abs_path(
                review_path, f'wakati_nva_lemma/{idx}.txt')
            # 分かち書きしたテキストをファイルへ出力
            with open(wakati_path, mode='w', encoding='utf_8') as f:
                for text in text_wakati:
                    # f.write('\n'.join(text))
                    f.write(text + '\n')
                print(f'file name "{idx}" was output. format: .txt')

    else:
        # 月別に改行したテキストを単一のファイルに出力する場合
        # 分かち書きしたテキストを出力するファイルのパスをセット
        wakati_path = pm.get_abs_path(review_path, 'wakati/all_wakati.txt')
        # ファイルを開く
        with open(wakati_path, mode='w', encoding='utf_8') as f:
            for idx, group in grouped:
                # 分かち書きしたコメントをリストへ格納
                text_wakati = ginza_de_wakatiko(group['comments'].values)
                # 分かち書きしたテキストをファイルへ出力
                for text in text_wakati:
                    # f.write('\n'.join(text))
                    f.write(text)
                else:
                    f.write('\n')
                print(f'"{idx}" wrote.')


def visualize_dep(text: str) -> None:
    """
    日本語文の文法的構造を解析し、可視化する。

    Parameters
    ----------
    text : str
        解析対象の日本語テキスト。

    Notes
    -----
    実行後、 ブラウザで http://localhost:5000 を開くと画像が表示される。
    """
    doc = nlp(text)
    options = {"compact": True, "bg": "#09a3d5",
               "color": "white", "font": "Source Sans Pro"}
    displacy.serve(doc, style="dep", options=options)


def visualize_dep_list():
    text = 'あいうえお。かきくけこ。さしすせそ'
    doc = nlp(text)
    sentence_spans = list(doc.sents)
    displacy.serve(sentence_spans, style="dep")


def visualize_ent():
    text = 'あのラーメン屋の油そばは絶品です。'
    doc = nlp(text)
    displacy.serve(doc, style="ent")


def save_as_image(text: str, path) -> None:
    """
    日本語文の文法的構造を解析し、結果を画像として保存する。

    Parameters
    ----------
    text : str
        解析対象の日本語テキスト。
    path
        保存先のファイルパス。

    Notes
    -----
    画像はSVG形式で保存される。
    """
    doc = nlp(text)
    svg = displacy.render(doc, style='dep')
    with open(path, mode='w') as f:
        f.write(svg)


def count_word(path: Path):

    # ファイルを開く
    with open(path, encoding='utf-8') as f:
        # ファイルの内容を読み出す
        data = f.read()
        # data = data.lower()  # 小文字にするならこのタイミングが楽
    counter = Counter(data.split())
    d = counter.most_common()
    return d[:30]


def test_bunsetsu(path: Path):

    df = pd.read_csv(path)
    df = df[df['rates'] < 3][:1]
    votes = df['votes'].apply(tf.extract_vote).values
    comments = df['comments'].values
    for comment, vote in zip(comments, votes):  # 一行ごとに
        try:
            doc = nlp(comment)  # 解析を実行し
        except Exception:
            continue
        for sent in doc.sents:  # 文単位でループ
            for t in br.bunsetu_head_tokens(sent):  # 文節主辞トークンのうち
                hoge = bunsetu_span(t)
                hoge = bunsetu_phrase_span(t)
                print(hoge)
                # if t.pos_ not in ["NOUN"]:
                #     continue  # 述語以外はスキップ
                # # if t.dep_ not in ["nsubj", 'obl']:
                # if t.dep_ not in ["nsubj"]:
                #     continue
                # tag = ginza.phrase(ginza.lemma_)(t)  # 述語とその格要素(主語・目的語相当)の句を集める
                # tag = tag.replace('+', '')
                # ent = ginza.bunsetu_span(t.head).text
                # print(f'{tag},{ent},{vote}')


def parse_document(sentence):

    doc = nlp(sentence)

    # # トークンのリストを生成
    # tokens = [token for token in doc]
    # # 主語述語ペアのリスト
    # subject_list = [f"{token.lemma_}:{tokens[token.head.i].lemma_}" for token in tokens if token.dep_ in [
    #     'nsubj', 'iobj']]
    # subject_list = [[f"{token.lemma_}:{tokens[token.head.i].lemma_}" for token in tokens if token.dep_ in ['nsubj', 'iobj']] for tokens in doc]

    # 主語述語ペアのリスト
    subject_list = [[token, doc[token.head.i]]
                    for token in doc if token.dep_ in ['nsubj', 'iobj', 'obl']]
    for idx, subject in enumerate(subject_list):
        if doc[subject[0].i-1].dep_ in ['compound', 'amod']:
            subject_list[idx][0] = doc[subject[0].i-1].text + \
                subject_list[idx][0].text
    # for idx, subject in enumerate(subject_list):
    #     if doc[subject[1].i+1].pos_ in ['AUX']:
    #         subject_list[idx][1] = subject_list[idx][1].text + \
    #             doc[subject[1].i+1].text
    for idx, subject in enumerate(subject_list):
        subject_list[idx][1] = subject_list[idx][1].text + \
            recursive_matching('pos', doc, subject, ['AUX'])

    # print(subject_list)
    # subject_list = [
    #     f"{token.lemma_}:{doc[token.head.i].lemma_}" for token in doc if token.dep_ in ['nsubj', 'iobj']]
    # subject_list = [
    #     f"{token.text}:{doc[token.head.i].text}" for token in doc if token.dep_ in ['nsubj', 'iobj', 'obl']]
    # subject_list = [
    #     f"{token.lemma_}:{doc[token.head.i].lemma_}" for token in doc if token.dep_ in ['aux']]

    return subject_list


def recursive_matching(mode: str, doc, subject: list,  pattern: list, i: int = 1):
    if mode == 'pos':
        if subject[1].i+i == len(doc):
            return ''
        if doc[subject[1].i+i].pos_ in pattern:
            # if doc[subject[1].i+i].pos_ in ['AUX']:
            if subject[1].i+i == len(doc) - 1:
                return doc[subject[1].i+i].text
            return doc[subject[1].i+i].text + recursive_matching(mode, doc, subject, pattern, i+1)
        else:
            return ''
    elif mode == 'dep':
        if subject[1].i+i == len(doc):
            return ''
        if doc[subject[1].i+i].pos_ in pattern:
            # if doc[subject[1].i+i].pos_ in ['AUX']:
            if subject[1].i+i == len(doc) - 1:
                return doc[subject[1].i+i].text
            return doc[subject[1].i+i].text + recursive_matching(mode, doc, subject, pattern, i+1)
        else:
            return ''


if __name__ == '__main__':
    test_texts = ['昨日は雨で洗濯物を干せませんでした。',
                  '路地裏の薄汚れた店舗ほど老舗感があるのは何故だろう。',
                  '天気予報では明日は晴れだ。てるてる坊主を逆さづりにして窓際に晒してやる。']
    # path = Path(
    #     'C:/Users/yaroy/Documents/Python_Projects/NLP/csv/reviews/B07VR7ZBBQ/wakati/2019-09-30.txt')
    # path = Path(
    #     'C:/Users/yaroy/Documents/Python_Projects/NLP/csv/reviews/B07VR7ZBBQ/wakati_nva_lemma/2019-10-31.txt')
    # path = Path(
    #     'C:/Users/yaroy/Documents/Python_Projects/NLP/csv/reviews/B07VR7ZBBQ/fB07VR7ZBBQ.csv')
    # path = Path(
    #     '/home/ubuntu/Documents/python_projects/py37/NLP/csv/reviews/B07VR7ZBBQ/fB07VR7ZBBQ.csv')
    path = Path('fearphone_review.csv')
    test_bunsetsu(path)

    # df = pd.read_csv(path)
    # print(df[df['rates'] < 3][:3])

    # output_comments_txt(path, is_div=True)

    # text = '購入後0週間程度は使えましたが充電してたにも関わらず電源が入らない充電もできなくなりました購入はおすすめしません'
    # subject_list = parse_document(text)
    # print(subject_list)
    # text = '安価な安っぽさも無く意外にも高級感のあるイヤホンでした同等金額のものをいくつも購入しましたが私はこのイヤホンが0番良かったです音もクリアで重厚感がありますお値段以上でとても良い買い物をしました'
    # text = 'お金も希望もない'
    # subject_list = parse_document(text)
    # print(subject_list)
    # visualize_dep(text)
    pass
