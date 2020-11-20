import fasttext as ft
import MeCab
import sys


def get_score(result, label_dict):
    scores = []
    for item in result:
        scores.append(label_dict[item[0].split(',')[0]]+':'+str(item[1]))
    return scores


def get_label_dict():
    dictionay = {'__label__1': '評価1', '__label__2': '評価2',
                 '__label__3': '評価3', '__label__4': '評価4', '__label__5': '評価5'}
    return dictionay


def get_token(content):
    tokens = []
    tagger = MeCab.Tagger('')
    tagger.parse('')
    node = tagger.parseToNode(content)
    while node:
        tokens.append(node.surface)
        node = node.next
    return tokens


def main(argv):
    model_name = argv[0]
    content = argv[1]
    # label_dict = get_label_dict()
    classifier = ft.load_model(model_name)
    tokens = get_token(content)
    estimate = classifier.predict([' '.join(tokens)], k=5)
    print(estimate)
    # scores = get_score(estimate[0], label_dict)
    # print(scores)


if __name__ == '__main__':
    main(sys.argv[1:])
