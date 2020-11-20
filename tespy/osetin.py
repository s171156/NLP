# -*- coding: utf-8 -*-
import os
import pathlib
import pandas as pd

import MeCab
import sengiri

NEGATION = ('ない', 'ず', 'ぬ')
PARELLEL_PARTICLES = ('か', 'と', 'に', 'も', 'や', 'とか', 'だの', 'なり', 'やら')


class Analyzer(object):

    def __init__(self, mecab_args='-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'):
        df = pd.read_csv(pathlib.Path('resources/sad/earphone.sad'))
        self.senti_dict = dict(zip(df['単語'], df['スコア']))
        self.tagger = MeCab.Tagger(mecab_args)
        self.tagger.parse('')  # for avoiding bug

    def _has_arujanai(self, substring):
        return 'あるじゃない' in substring

    def _calc_sentiment_polarity(self, sentence):
        polarities = []
        lemmas = []
        n_parallel = 0
        substr_count = 0
        node = self.tagger.parseToNode(sentence)
        while node:
            if 'BOS/EOS' not in node.feature:
                surface = node.surface
                substr_count += len(surface)
                feature = node.feature.split(',')
                lemma = feature[6] if feature[6] != '*' else node.surface
                if lemma in self.senti_dict:
                    polarity = 1 if self.senti_dict[lemma] > 0 else -1
                    n_parallel += node.next.surface in PARELLEL_PARTICLES
                else:
                    polarity = None
                if polarity:
                    polarities.append([lemma, polarity])
                elif polarities and surface in NEGATION and not self._has_arujanai(sentence[:substr_count]):
                    polarities[-1][1] *= -1
                    if polarities[-1][0].endswith('-NEGATION'):
                        polarities[-1][0] = polarities[-1][0][:-9]
                    else:
                        polarities[-1][0] += '-NEGATION'
                    # parallel negation
                    if n_parallel and len(polarities) > 1:
                        n_parallel = len(polarities) if len(
                            polarities) > n_parallel else n_parallel + 1
                        n_parallel = n_parallel + \
                            1 if len(polarities) == n_parallel else n_parallel
                        for i in range(2, n_parallel):
                            polarities[-i][1] *= -1
                            if polarities[-i][0].endswith('-NEGATION'):
                                polarities[-i][0] = polarities[-i][0][:-9]
                            else:
                                polarities[-i][0] += '-NEGATION'
                        n_parallel = 0
                lemmas.append(lemma)
            node = node.next
        return polarities

    def count_polarity(self, text):
        """Calculate sentiment polarity counts per sentence
        Arg:
            text (str)
        Return:
            counts (list) : positive and negative counts per sentence
        """
        counts = []
        for sentence in sengiri.tokenize(text):
            count = {'positive': 0, 'negative': 0}
            polarities = self._calc_sentiment_polarity(sentence)
            for polarity in polarities:
                if polarity[1] == 1:
                    count['positive'] += 1
                elif polarity[1] == -1:
                    count['negative'] += 1
            counts.append(count)
        return counts

    def analyze(self, text):
        """Calculate sentiment polarity scores per sentence
        Arg:
            text (str)
        Return:
            scores (list) : scores per sentence
        """
        scores = []
        for sentence in sengiri.tokenize(text):
            polarities = self._calc_sentiment_polarity(sentence)
            if polarities:
                scores.append(sum(p[1] for p in polarities) / len(polarities))
            else:
                scores.append(0)
        return scores

    def analyze_detail(self, text):
        """Calculate sentiment polarity scores per sentence
        Arg:
            text (str)
        Return:
            results (list) : analysis results
        """
        results = []
        for sentence in sengiri.tokenize(text):
            polarities = self._calc_sentiment_polarity(sentence)
            if polarities:
                result = {
                    'positive': [p[0] for p in polarities if p[1] == 1],
                    'negative': [p[0] for p in polarities if p[1] == -1],
                    'score': sum(p[1] for p in polarities) / len(polarities),
                }
            else:
                result = {'positive': [], 'negative': [], 'score': 0.0}
            results.append(result)
        return results


if __name__ == "__main__":
    analy = Analyzer()
    score = analy.analyze_detail(
        '音漏れが酷い。使い物にならない。')
    print(score)
