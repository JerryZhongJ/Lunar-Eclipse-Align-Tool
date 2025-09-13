# main.py
import tkinter as tk
from ui import UniversalLunarAlignApp
from utils_common import force_garbage_collection
try:
    from ttkthemes import ThemedTk
except Exception:
    ThemedTk = None

import platform
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"

# 本代码由 @正七价的氟离子 原始创建，ChatGPT、Manus AI、Claude优化与注释
# 感谢南京大学@黄喵 帮忙修改代码，﻿@黑灯kuro﻿ 提出想法，﻿@无尽碗莲﻿ ﻿@摄影师无敌武士兔﻿ @飞翔的荷兰者 李老师 ﻿@UIN丨huaji﻿ 提供素材和图片用于测试

def main():
    try:
        if ThemedTk is not None:
            root = ThemedTk(theme=("winnative" if IS_WINDOWS else "aqua" if IS_MACOS else "arc"))
        else:
            raise ImportError("ttkthemes not available")
    except Exception as e:
        print(f"主题加载失败，使用默认样式: {e}")
        root = tk.Tk()

    app = UniversalLunarAlignApp(root)

    def on_closing():
        force_garbage_collection()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        force_garbage_collection()

# Note: ROI refine is now translation-only (no rotation).
# 结果输出/日志中应使用“平移对齐”，不再显示“θ=...”或“旋转”
# 例如: print(f"Multi-ROI refine (Translation-only, inliers=..., roi=..., ...s)")

if __name__ == "__main__":
    main()
