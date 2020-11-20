import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ReviewPlotter:

    def __init__(self, headless: bool = False):
        if headless:
            matplotlib.use('Agg')
        self.fig = plt.figure()
        self.fig.subplots_adjust(bottom=0.2)
        self.ax = [None, None]

    def plot_stacked_bar_graph(self, df: pd.DataFrame, ratio: bool = False, rates: list = None, title: str = None):
        '''
        Amazonレビューの時系列データを積み上げ棒グラフで表示します。
        '''
        self.set_axis(layer=0)
        left = np.arange(len(df.index))
        self.ax[0].set_title(title)
        self.ax[0].set_xlabel('Period')
        self.ax[0].set_ylabel('Freq')
        self.ax[0].grid()
        self.ax[0].set_axisbelow(True)
        self.ax[0].set_xticks(left)
        self.ax[0].set_xticklabels(df.index, rotation=60)
        color_list = ['g', 'c', 'b', 'm', 'r']
        bottoms = 0
        if rates is None:
            rates = list(range(1, 6))
            # rates.sort(reverse=True)
        for rate in rates:
            if ratio:
                self.ax[0].bar(left, df[rate]/df[15], align='center', label=rate,
                               bottom=bottoms, color=color_list[rate-1])
                bottoms += df[rate]/df[15]
            else:
                self.ax[0].bar(left, df[rate], align='center', label=rate,
                               bottom=bottoms, color=color_list[rate-1])
                # オーバーフロー対策
                bottoms += df[rate].astype('int16')
            self.ax[0].legend()

    def plot_histgram_of_rate(self, df: pd.DataFrame):
        '''
        Amazonレビューの評価をヒストグラムにプロットします。
        '''
        self.set_axis(layer=0)
        # ビン数をセット
        self.ax[0].hist(df['rates'], bins=5, range=(1, 6))
        # ビンの間隔をセット
        self.ax[0].set_xticks([1.5, 2.5, 3.5, 4.5, 5.5])
        # x軸の目盛りのラベルをセット
        self.ax[0].set_xticklabels(range(1, 6))
        # プロットのタイトルをセット
        self.ax[0].set_title("Customer Review's Rate")
        # x軸のラベルをセット
        self.ax[0].set_xlabel('Rate')
        # y軸のラベルをセット
        self.ax[0].set_ylabel('Freq')

    def plot_line_of_rate(self, df: pd.DataFrame, ratio: bool = False, rates: list = None, title: str = None):
        '''
        Amazonレビューを折れ線グラフにプロットします。
        '''
        self.set_axis(layer=1)
        # 抽出したい評価に指定がない場合は全ての評価をセット
        if rates is None:
            rates = range(1, 6)
        left = np.arange(len(df.index))
        # self.ax[1].grid()
        self.ax[1].set_xticks(left)
        self.ax[1].set_xticklabels(df.index, rotation=60)
        self.ax[1].set_title(title)
        for rate in rates:
            if ratio:
                self.ax[1].plot(
                    left, df[rate]/df[15], linestyle='solid', marker='o', label=rate)
            else:
                self.ax[1].plot(
                    left, df[rate], linestyle='solid', marker='o', label=rate)

            self.ax[1].legend()

    def set_axis(self, layer: int):
        '''
        Axisオブジェクトをセットします。
        '''

        if layer == 0:
            if self.ax[0] is None:
                if self.ax[1] is None:
                    self.ax[0] = self.fig.add_subplot(111)
                else:
                    raise OverPlotError('[Layer: 1] グラフの上書きが実行されました。')
                    # self.ax[0] = self.ax[1].twinx()
                    # pass
            else:
                raise OverPlotError('[Layer: 0] グラフの上書きが実行されました。')
        elif layer == 1:
            if self.ax[1] is None:
                if self.ax[0] is None:
                    self.ax[1] = self.fig.add_subplot(111)
                else:
                    self.ax[1] = self.ax[0].twinx()
            if self.ax[0] is not None:
                self.ax[1].patch.set_alpha(0)

    @staticmethod
    def plot():
        plt.show()

    @staticmethod
    def save(path_plt):
        plt.savefig(path_plt)
        plt.close()


class OverPlotError(Exception):
    '''
    Exception when graph overwriting occurs by matplotlib
    '''
    pass


def test_plot():
    # fig = plt.figure()
    # fig.subplots_adjust(bottom=0.2)
    # ax = fig.add_subplot(111)
    # # fig.grid(which='major')
    # left = np.arange(len(df.index))
    # ax.set_xticks(left)
    # ax.set_xticklabels(df.index, rotation=60)
    # rate = 1
    # ax.plot(left, df[rate], linestyle='solid', marker='o', label=rate)
    # ax.grid()
    # # ax.set_facecolor('m')
    # # ax.patch.set_alpha(0)
    # rate = 2
    # ax.plot(left, df[rate], linestyle='solid', marker='o', label=rate)
    # ax.legend()
    # plt.show()
    pass


if __name__ == "__main__":
    pass
