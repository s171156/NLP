import time


def measure_processing_speed(func):
    '''
    関数の実行速度を計測します。
    '''
    def wrapper(*args, **kwargs):
        # 処理の開始時刻
        start = time.time()
        # 関数の実行
        res = func(*args, **kwargs)
        # 処理の終了時刻
        end = time.time()
        # 処理時間
        elapsed_time = end - start
        # 処理時間の表示
        print(f"elapsed_time: {elapsed_time} [sec]")
        return res
    return wrapper
