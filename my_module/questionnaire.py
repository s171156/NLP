import sys
import re


class Questionnaire(object):

    def ask_y_or_n(self) -> bool:
        '''
        二者択一（YES/NO）の入力を要求します。
        '''
        while True:
            answer = input().strip()
            if answer == 'y':
                return True
            elif answer == 'n':
                return False
            elif answer == 'q':
                print("'q'の入力を検出したためプログラムを終了します。")
                sys.exit()

    def ask_keyword(self) -> str:
        '''
        任意の文字列の入力を要求します。
        '''
        while True:
            keyword = input().strip()
            if keyword == 'q':
                print("'q'の入力を検出したためプログラムを終了します。")
                sys.exit()
            elif not keyword:
                print('キーワードを入力してください')
            elif keyword:
                print(f"入力は'{keyword}'で間違いありませんか？     :   y/n")
                if self.ask_y_or_n():
                    return keyword

    def ask_number(self) -> str:
        '''
        数字の入力を要求します。
        '''
        while True:
            number = input().strip()
            if number == 'q':
                print("'q'の入力を検出したためプログラムを終了します。")
                sys.exit()
            elif not number:
                print('全角または半角アラビア数字を入力してください')

            if number.isdecimal() is False:
                print('全角または半角アラビア数字を入力してください')
                continue
            print(f"入力は'{number}'で間違いありませんか？     :   y/n")
            if self.ask_y_or_n():
                return int(number)

    def ask_url(self) -> str:
        '''
        URLの入力を要求します。
        '''

        # https or http pattern
        pattern = 'http(s*)://'
        repatter = re.compile(pattern)

        while True:
            url = input().strip()
            if url == 'q':
                print("'q'の入力を検出したためプログラムを終了します。")
                sys.exit()
            elif not url:
                print('URLを入力してください。')
            elif url:
                if re.search(repatter, url) is False:
                    print('不正なURLです。')
                    print("'http'もしくは'https'から始まるURLを入力してください。")
                    continue
                print(f"'{url}'で間違いありませんか？     :   y/n")
                if self.ask_y_or_n():
                    return url

    def show_dict(self, _dict: dict):
        '''
        辞書の内容を確認します。
        '''
        while True:
            print()
            print('=' * 50)
            print('Parameters'.center(50))
            for key, value in _dict.items():
                print(f'{key}'.ljust(25), end='')
                print(':', end='')
                print(f'{value}'.rjust(24))
            print('=' * 50)
            print()

            print('上記の内容でよろしいでしょうか？       :   y/n')
            return self.ask_y_or_n()


if __name__ == "__main__":
    pass
