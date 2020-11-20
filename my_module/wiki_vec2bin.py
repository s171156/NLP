import gensim
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
    prog='wiki_vec2bin.py',
    usage='wiki_vec2binですが、fastTextのモデルならOK！',
    description='fastTextが標準で生成する.binファイルに修正を施します。',
    add_help=True
)
parser.add_argument('src', help='fastTextモデルのパス　例）hoge.vec')
args = parser.parse_args()

if __name__ == "__main__":
    src = args.src
    src = Path(src)
    dst = src.with_name(f'{src.stem}_fixed.bin')
    # model.vecをload
    model = gensim.models.KeyedVectors.load_word2vec_format(
        src, binary=False)

    # バイナリファイルとして保存
    model.save_word2vec_format(dst, binary=True)
    pass
