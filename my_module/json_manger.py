from pathlib import Path
import json


def generate_json(file_path: str, dict_json: dict):
    '''
    辞書のリスト型jsonファイルを生成

    Args
    ----
    file_path   :   str
        出力先のパス
    dict_json   :   dict
        記述したい辞書型データ

    Notes
    -----
    同名の既存jsonファイルには末尾にデータを挿入する。
    '''

    # jsonデータを生成する
    with Path(file_path).open(mode='ab+') as f:
        f.seek(0, 2)                                    # ファイル末尾に移動
        if f.tell() == 0:                               # ファイルが空かチェック
            f.write('[\n'.encode())
            f.write(json.dumps(dict_json, indent=4,
                               ensure_ascii=False
                               ).encode())
            f.write('\n]'.encode())
        else:
            f.seek(-1, 2)                               # ファイル末尾から-1文字移動
            f.truncate()                                # 最後の文字(])を削除し、jsonの配列を開く
            f.seek(-1, 2)
            f.truncate()
            f.write(',\n'.encode())                     # 最後のセパレーターを書き込む

            f.write(json.dumps(dict_json, indent=4,
                               ensure_ascii=False
                               ).encode())              # 辞書をjson形式でダンプ書き込み
            f.write('\n]'.encode())                     # jsonの配列を閉じる


def append_json_to_file(data: dict, path_file: str) -> bool:
    with open(path_file, 'ab+') as f:              # ファイルを開く
        f.seek(0, 2)                                # ファイルの末尾（2）に移動（フォフセット0）
        if f.tell() == 0:                         # ファイルが空かチェック
            f.write(json.dumps([data]).encode())   # 空の場合は JSON 配列を書き込む
        else:
            f.seek(-1, 2)                           # ファイルの末尾（2）から -1 文字移動
            f.truncate()                           # 最後の文字を削除し、JSON 配列を開ける（]の削除）
            f.write(' , '.encode())                # 配列のセパレーターを書き込む
            f.write(json.dumps(data).encode())     # 辞書を JSON 形式でダンプ書き込み
            f.write(']'.encode())                  # JSON 配列を閉じる
    return f.close()  # 連続で追加する場合は都度 Open, Close しない方がいいかも
