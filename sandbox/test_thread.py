import threading
import time

def worker():
    print("スレッド開始")
    time.sleep(2)
    print("スレッド終了")

# --- スレッド未生成・未起動 ---
t = None

# まだスレッドは無いので is_alive 判定できない
print("1: t is None?", t is None)  # True

# スレッド生成 & 起動
t = threading.Thread(target=worker)
print("2: 起動前 is_alive:", t.is_alive())  # False

t.start()  # ここでworker()が別スレッドで動く

print("3: 起動直後 is_alive:", t.is_alive())  # True

# 少し待つ（workerがまだ動いている）
time.sleep(1)
print("4: 1秒後 is_alive:", t.is_alive())  # True

# 終了待ち
t.join()
print("5: 終了後 is_alive:", t.is_alive())  # False
