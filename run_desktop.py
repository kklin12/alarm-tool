import os
import sys
import threading
import webbrowser
import time

def fix_runtime_path():
    if hasattr(sys, "_MEIPASS"):
        os.chdir(os.path.dirname(sys.executable))

fix_runtime_path()

from app import app

def start_server():
    try:
        # 这里注意：debug=False，use_reloader=False
        app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)
    except Exception as e:
        print(f"启动失败: {e}")
        time.sleep(10)

def open_browser():
    time.sleep(1.5)
    # 这里同步改成5001端口
    webbrowser.open("http://127.0.0.1:5001")

if __name__ == "__main__":
    print("=" * 60)
    print(" ✈️ 飞机音响告警生成与评估工具 正在启动...")
    print(" = 请勿关闭此窗口 =")
    print("=" * 60)

    threading.Thread(target=start_server, daemon=True).start()
    threading.Thread(target=open_browser, daemon=True).start()

    while True:
        try:
            time.sleep(3600)
        except KeyboardInterrupt:
            break

print("程序已关闭")