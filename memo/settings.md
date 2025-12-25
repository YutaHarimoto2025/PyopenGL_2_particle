❯ pip show pyQt6
Name: PyQt6
Version: 6.10.1
でバージョン固定

```
# PyQt6のライブラリ不整合を解消するための設定
export LD_LIBRARY_PATH="/home/yuta-harimoto/.home_env/lib/python3.12/site-packages/PyQt6/Qt6/lib:$LD_LIBRARY_PATH"
```