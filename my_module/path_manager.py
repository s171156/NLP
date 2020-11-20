from pathlib import Path


def get_abs_path(basis: Path, addnl: Path = '') -> Path:
    '''
    基点のパスから追加（目的まで）のパスを連結した絶対パスを取得します。

    Args
    ----
    basis: Path
        基点パス
    addnl: str
        追加パス

    Returns
    -------
    連結パス（基点パス/追加パス）の絶対パスを返します。

    Notes
    -----
    パスの最後がファイルの場合は親ディレクトリまでのディレクトリを作成します。
    '''

    # '/'が先頭の追加パスに'joinpath'をした場合に、'C:'に追加パスが連結される挙動を防ぐ
    if addnl.startswith('/'):
        # 先頭の'/'を除去する
        addnl = addnl[1:]

    # パスの末尾がファイルであるか
    # 'if_file()'は既存のファイルの有無の確認。'suffix'拡張子の有無でパスの末尾がファイルか判定する。
    if Path(basis).suffix:
        # パスの末尾がファイルであるか
        if Path(addnl).suffix:
            # 親ディレクトリまでのディレクトリを生成
            Path(basis).resolve().parent.joinpath(addnl).parent.mkdir(
                parents=True, exist_ok=True)
        else:
            # ディレクトリを生成
            Path(basis).resolve().parent.joinpath(addnl).mkdir(
                parents=True, exist_ok=True)

        # 連結パスの絶対パス
        return Path(basis).resolve().parent.joinpath(addnl)
    else:
        # パスの末尾がファイルであるか
        if Path(addnl).suffix:
            # 親ディレクトリまでのディレクトリを生成
            Path(basis).resolve().joinpath(addnl).parent.mkdir(
                parents=True, exist_ok=True)
        else:
            # ディレクトリを生成
            Path(basis).resolve().joinpath(addnl).mkdir(
                parents=True, exist_ok=True)

        # 連結パスの絶対パス
        return Path(basis).resolve().joinpath(addnl)


if __name__ == "__main__":
    pass
