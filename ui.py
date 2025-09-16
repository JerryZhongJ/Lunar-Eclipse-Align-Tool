# ui.py
import os, tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
import cv2
import threading, queue, time

from utils_common import (
    VERSION, SYSTEM, DEFAULT_DEBUG_MODE, SUPPORTED_EXTS, UI_FONT, DEFAULT_FONT,
    normalize_path, safe_join, imread_unicode, to_display_rgb,
    force_garbage_collection, log
)

from pipeline import align_moon_images_incremental

import algorithms_circle as _algo_circle
class DebugWindow(tk.Toplevel):
    """独立的调试窗口：可选择样张并实时调节 param1/param2/半径范围，
    同时可叠加显示"分析区域"掩膜与检测到的圆/圆心标注。
    该窗口不会改变主流程的检测逻辑，仅用于参数可视化调节。
    若勾选"启用调试输出"，会把当前样张路径与开关写回主界面变量，
    以便 pipeline 在处理时输出相应的调试图像。
    """
    def __init__(self, master, app_controller):
        super().__init__(master)
        self.app = app_controller
        self.title("调试窗口（参数实时预览）")
        self.geometry("980x680"); self.minsize(760, 520)
        self.preview_img_cv = None  # BGR (cv2 读取)
        self.preview_rgb = None     # RGB 显示
        self.preview_base_rgb = None  # 原始RGB(不带任何叠加)
        self.preview_gray_rgb = None  # 灰度RGB(用于叠加显示)
        self.preview_disp = None
        self.preview_scale = 1.0
        self.current_path = None

        # 调试计算的后台执行控制
        self._dbg_queue = queue.Queue()
        self._dbg_worker = None
        self._dbg_cancel = threading.Event()
        self._dbg_job_id = 0
        self._dbg_busy = False
        self._dbg_pending = False
        self._last_det = None  # 最近一次检测结果 (dict 或 None)

        # 参数（默认取主界面当前值）
        p = self.app.params
        self.min_r = tk.IntVar(value=p["min_radius"].get())
        self.max_r = tk.IntVar(value=p["max_radius"].get())
        self.param1 = tk.IntVar(value=p["param1"].get())
        self.param2 = tk.IntVar(value=p["param2"].get())
        self.show_mask = tk.BooleanVar(value=False)
        self.enable_debug = tk.BooleanVar(value=self.app.debug_var.get())
        self.enable_debug.trace_add('write', lambda *a: (self._sync_debug_back(), self.refresh()))

        # --- Helper callables for robust algorithms_circle access ---
        import numpy as np
        def _call_build_mask(this, gray):
            try:
                # Try UI version first
                try:
                    return _algo_circle.build_analysis_mask_ui(gray, brightness_min=3/255.0)
                except AttributeError:
                    pass
                # Try new signature
                try:
                    return _algo_circle.build_analysis_mask(gray, brightness_min=3/255.0)
                except TypeError:
                    # Old signature: no brightness_min
                    return _algo_circle.build_analysis_mask(gray)
            except Exception:
                # Fallback: return zeros mask
                return np.zeros_like(gray, dtype='uint8')

        def _call_debug_detect(this, img_bgr, min_r, max_r, p1, p2):
            # Try debug_detect_circle (positional), then (keyword), then debug_detect
            try:
                try:
                    return _algo_circle.debug_detect_circle(img_bgr, min_r, max_r, p1, p2)
                except TypeError:
                    try:
                        return _algo_circle.debug_detect_circle(img_bgr, min_radius=min_r, max_radius=max_r, param1=p1, param2=p2)
                    except Exception:
                        return _algo_circle.debug_detect(img_bgr, min_r, max_r, p1, p2)
            except Exception:
                return None

        self._call_build_mask = _call_build_mask.__get__(self)
        self._call_debug_detect = _call_debug_detect.__get__(self)
        self._build_ui()
        self.center()
        self.after(40, self._poll_debug_results)

    def center(self):
        self.update_idletasks()
        w,h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth()//2) - (w//2)
        y = (self.winfo_screenheight()//2) - (h//2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.rowconfigure(2, weight=1)  # 修改为第2行（画布行）
        root.columnconfigure(0, weight=1)

        # 第一行工具条
        bar1 = ttk.Frame(root)
        bar1.grid(row=0, column=0, sticky="ew", pady=(0,4))
        ttk.Button(bar1, text="选择样张", command=self.choose_image).pack(side="left")

        ttk.Label(bar1, text="最小半径").pack(side="left", padx=(12,4))
        ttk.Spinbox(bar1, from_=1, to=4000, textvariable=self.min_r, width=6, command=self.refresh).pack(side="left")
        ttk.Label(bar1, text="最大半径").pack(side="left", padx=(12,4))
        ttk.Spinbox(bar1, from_=1, to=5000, textvariable=self.max_r, width=6, command=self.refresh).pack(side="left")
        ttk.Label(bar1, text="参数1").pack(side="left", padx=(12,4))
        ttk.Spinbox(bar1, from_=1, to=200, textvariable=self.param1, width=5, command=self.refresh).pack(side="left")
        ttk.Label(bar1, text="参数2").pack(side="left", padx=(12,4))
        ttk.Spinbox(bar1, from_=1, to=100, textvariable=self.param2, width=5, command=self.refresh).pack(side="left")

        # 第二行工具条
        bar2 = ttk.Frame(root)
        bar2.grid(row=1, column=0, sticky="ew", pady=(0,8))
        
        ttk.Checkbutton(bar2, text="显示分析区域", variable=self.show_mask, command=self.refresh).pack(side="left")
        self.use_pipeline_algo = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar2, text="用主流程算法", variable=self.use_pipeline_algo, command=lambda: (self._clear_last_det(), self.refresh(), self._schedule_debug_compute())).pack(side="left", padx=(10,0))
        ttk.Checkbutton(bar2, text="启用调试输出", variable=self.enable_debug, command=lambda: (self._sync_debug_back(), self.refresh())).pack(side="left", padx=(10,0))

        # 画布
        self.canvas = tk.Canvas(root, background="#222", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")

        # 绑定变量变化自动刷新
        for var in (self.min_r, self.max_r, self.param1, self.param2):
            var.trace_add('write', lambda *args: (self.refresh(), self._schedule_debug_compute()))

    def choose_image(self):
        initdir = self.app.input_var.get() if os.path.isdir(self.app.input_var.get()) else os.getcwd()
        filetypes = [("支持的图像", " ".join(f"*{ext}" for ext in SUPPORTED_EXTS)), ("所有文件", "*.*")]
        path = filedialog.askopenfilename(title="选择调试样张", filetypes=filetypes, initialdir=initdir, parent=self)
        if not path:
            return
        self.current_path = normalize_path(path)
        img = imread_unicode(self.current_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("错误", "无法读取该图像。", parent=self)
            return
        self.preview_img_cv = img
        self.preview_rgb = to_display_rgb(img)
        self.preview_base_rgb = self.preview_rgb.copy()
        # 生成灰度版（用于叠加显示时，先把原图变黑白）
        try:
            if img.ndim == 3:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                g = img.copy()
            g_bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            self.preview_gray_rgb = to_display_rgb(g_bgr)
        except Exception:
            # 兜底：若转换失败则退回彩色
            self.preview_gray_rgb = self.preview_base_rgb.copy()
        # 将样张路径写回主界面（供 pipeline 输出调试文件时使用）
        self.app.debug_image_path_var.set(self.current_path)
        self.enable_debug.set(True)
        self._sync_debug_back()
        self.refresh()
        self._schedule_debug_compute()

    def _draw_image(self, rgb=None):
        self.canvas.delete("all")
        img_rgb = rgb if rgb is not None else self.preview_rgb
        if img_rgb is None:
            self.canvas.update_idletasks()
            cw,ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            self.canvas.create_text(cw/2, ch/2, text="请选择一张样张…", fill="gray70")
            return
        h,w = img_rgb.shape[:2]
        cw,ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.preview_scale = min(cw/max(w,1), ch/max(h,1), 1.0)
        nw,nh = int(w*self.preview_scale), int(h*self.preview_scale)
        disp = Image.fromarray(img_rgb).resize((nw, nh), Image.LANCZOS)
        self.preview_disp = ImageTk.PhotoImage(disp)
        self.canvas.create_image(cw//2, ch//2, image=self.preview_disp, anchor="center")

    def _sync_debug_back(self):
        # 把开关写回主界面，供 pipeline 读取，同时用于本窗口叠加圆/圆心的显示总开关
        en = bool(self.enable_debug.get())
        self.app.debug_var.set(en)

    def _clear_last_det(self):
        self._last_det = None

    def _detect_best(self, img_bgr):
        """Run circle detection with current sliders. If not found, try
        progressively wider & more tolerant UI‑only fallbacks. This does not
        change the pipeline behavior; it only makes the preview robust.
        Returns a tuple: (det_dict_or_None, used_fallback: bool).
        """
        import numpy as _np
        import cv2 as _cv2

        # helper: score candidate circles by edge response
        def _score_by_edges(gray, circles):
            if circles is None:
                return None
            edges = _cv2.Canny(gray, 60, 120)
            best = None; best_score = -1e9
            for x, y, r in circles:
                x, y, r = float(x), float(y), float(r)
                theta = _np.linspace(0, 2*_np.pi, 360, endpoint=False)
                xs = _np.clip(_np.round(x + r*_np.cos(theta)).astype(_np.int32), 0, gray.shape[1]-1)
                ys = _np.clip(_np.round(y + r*_np.sin(theta)).astype(_np.int32), 0, gray.shape[0]-1)
                score = edges[ys, xs].sum()
                if score > best_score:
                    best_score = score; best = (x, y, r)
            return best

        try:
            min_r = int(self.min_r.get()); max_r = int(self.max_r.get())
            p1 = int(self.param1.get()); p2 = int(self.param2.get())
        except Exception:
            min_r, max_r, p1, p2 = 10, 9999, 50, 30

        # --- Thumbnail downscaling (like pipeline) ---
        MAX_SIDE = 1600
        img = img_bgr
        H0, W0 = img.shape[:2]
        s = max(H0, W0) / float(MAX_SIDE)
        if s > 1.0:
            Hs = int(round(H0 / s))
            Ws = int(round(W0 / s))
            bgr_small = _cv2.resize(img, (Ws, Hs), interpolation=_cv2.INTER_AREA)
        else:
            s = 1.0
            bgr_small = img.copy()
            Hs, Ws = H0, W0

        # scale radii to thumbnail
        min_r_s = int(round(min_r / s))
        max_r_s = int(round(max_r / s))
        min_r_s = max(1, min_r_s)
        max_r_s = max(min_r_s+1, max_r_s)

        # --- Pass 1: use algorithms_circle.debug_detect_circle on thumbnail ---
        try:
            det = self._call_debug_detect(bgr_small, min_r_s, max_r_s, p1, p2)
            if det is not None and det.get('circle') is not None:
                if 'proc_shape' not in det:
                    det['proc_shape'] = (Hs, Ws)
                return det, False
        except Exception:
            det = None

        # Prepare grayscale from bgr_small
        try:
            gray = _cv2.cvtColor(bgr_small, _cv2.COLOR_BGR2GRAY) if bgr_small.ndim == 3 else bgr_small.copy()
        except Exception:
            return None, False

        # Light preprocessing similar to pipeline thumbnail path
        try:
            clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            g_small = clahe.apply(gray)
        except Exception:
            g_small = gray
        g_small = _cv2.GaussianBlur(g_small, (3, 3), 0)

        H, W = g_small.shape[:2]; minDim = min(H, W)

        # --- Pass 2: user range as-is (UI quick test) ---
        rngs = [(max(5, min_r_s), max(6, max_r_s))]
        # --- Pass 3: relaxed user range ---
        lo_rel = max(5, int(min(min_r_s, max_r_s) * 0.6))
        hi_rel = min(int(max(min_r_s, max_r_s) * 2.6), int(minDim/2))
        rngs.append((lo_rel, max(hi_rel, lo_rel+1)))
        # --- Pass 4: image‑proportional auto range ---
        rngs.append((max(8, int(minDim*0.08)), int(minDim*0.49)))

        p_sets = [(p1, p2), (max(30, p1), max(12, int(p2*0.7))), (80, 25), (60, 20)]

        for i, (lo, hi) in enumerate(rngs):
            lo = max(5, min(lo, hi-1))
            minDist = max(30, minDim // 4)
            for (pp1, pp2) in p_sets:
                try:
                    circles = _cv2.HoughCircles(g_small, _cv2.HOUGH_GRADIENT, dp=1.2, minDist=minDist,
                                                param1=int(pp1), param2=int(pp2),
                                                minRadius=int(lo), maxRadius=int(hi))
                except Exception:
                    circles = None
                if circles is None:
                    continue
                circles = _np.squeeze(circles, axis=0)
                best = _score_by_edges(g_small, circles)
                if best is not None:
                    x, y, r = best
                    used_fb = (i != 0)
                    return {"circle": (float(x), float(y), float(r)),
                            "quality": None,
                            "proc_shape": (Hs, Ws),
                            "used_fallback": used_fb}, used_fb

        return None, True

    def refresh(self):
        """Re-render preview according to current toggles/params.
        非阻塞：不在 UI 线程做重计算。只负责绘图，并触发后台计算。
        """
        # 选择底图：若勾选分析区域或启用调试输出，则用灰度底图；否则用原图
        base = None
        want_gray = bool(self.show_mask.get()) or bool(self.enable_debug.get())
        src_rgb = self.preview_gray_rgb if want_gray else self.preview_base_rgb
        if src_rgb is not None:
            base = src_rgb.copy()
        if base is None:
            self._draw_image()
            return

        disp_rgb = base.copy()
        # 不勾选时确保不保留之前的红色叠加（每次都从 base 复制绘制）
        if self.show_mask.get():
            try:
                gray = (cv2.cvtColor(self.preview_img_cv, cv2.COLOR_BGR2GRAY)
                        if (self.preview_img_cv.ndim == 3) else self.preview_img_cv)
                mask = self._call_build_mask(gray)
                overlay = disp_rgb
                H, W = overlay.shape[:2]
                m = (mask * 255).astype('uint8')
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                red = overlay.copy(); red[..., 0] = 255; red[..., 1] = 0; red[..., 2] = 0
                alpha = (m/255.0)[..., None] * 0.35
                disp_rgb = (overlay*(1-alpha) + red*alpha).astype('uint8')
            except Exception as e:
                print(f"显示分析区域失败: {e}")
                disp_rgb = base.copy()

        # 3) 先画图（会设置 self.preview_scale）
        self._draw_image(disp_rgb)

        # 4) 若未启用调试覆盖，只显示提示
        if not bool(self.enable_debug.get()):
            self.canvas.create_text(10, 10, anchor='nw', fill='white',
                                    text="调试未启用（仅显示原图）")
            return

        # 5) 叠加最近一次检测结果（如果有）
        det = self._last_det
        status_text = ""
        # 1) 有圆 -> 画圆并显示质量
        if det and isinstance(det, dict) and det.get('circle') is not None:
            q = det.get('quality', None)
            cx, cy, r = det['circle']
            try:
                sx, sy = 1.0, 1.0
                if 'proc_shape' in det:
                    ph, pw = det['proc_shape']
                    H0, W0 = self.preview_img_cv.shape[:2]
                    if pw > 0 and ph > 0:
                        sx = W0 / float(pw); sy = H0 / float(ph)
                elif 'scale' in det:
                    s = float(det['scale'])
                    sx = sy = 1.0 / s if s > 0 else 1.0
                cx, cy, r = cx * sx, cy * sy, r * sx
            except Exception:
                pass

            H, W = disp_rgb.shape[:2]
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            nw, nh = int(W * self.preview_scale), int(H * self.preview_scale)
            ox, oy = (cw - nw) // 2, (ch - nh) // 2
            s = self.preview_scale
            x, y, rr = ox + cx * s, oy + cy * s, r * s
            self.canvas.create_oval(x - rr, y - rr, x + rr, y + rr, outline="#FF4D4F", width=2)
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#FF4D4F", outline="")
            status_text = f"检测到圆 r≈{r:.1f}px" + ("  (备用范围)" if det.get('used_fallback') else "")
            if q is not None:
                status_text += f"  quality={q:.2f}"
        else:
            # 2) 无圆，但处于计算/等待阶段
            if self._dbg_busy or self._dbg_pending:
                status_text = "未检测到，正在检测…"
            else:
                # 3) 无圆，且没有在计算 -> 说明本次检测已结束
                status_text = "检测完成，未检测到圆"

        if status_text:
            self.canvas.create_text(10, 10, anchor='nw', fill='white', text=status_text)

        # 6) 触发一次后台计算（防抖）
        # （已移除：不再每次refresh都自动_schedule_debug_compute，避免重复计算）


    def _schedule_debug_compute(self):
        """标记需要计算；若当前不忙立即启动一次。"""
        self._dbg_pending = True
        # 先刷新一次，让状态文本显示"正在检测…"
        self.refresh()
        if not self._dbg_busy and self.preview_img_cv is not None and bool(self.enable_debug.get()):
            self._start_debug_compute()

    def _start_debug_compute(self):
        """启动后台线程做一次检测（使用 _detect_best，不阻塞 UI）。"""
        if self.preview_img_cv is None:
            return
        self._dbg_busy = True
        self._dbg_pending = False
        self._dbg_job_id += 1
        job_id = self._dbg_job_id
        img = self.preview_img_cv.copy()

        # 读取当前 UI 参数
        try:
            min_r = int(self.min_r.get()); max_r = int(self.max_r.get())
            p1 = int(self.param1.get()); p2 = int(self.param2.get())
        except Exception:
            min_r, max_r, p1, p2 = 10, 9999, 50, 30

        # 取消上一个（结果会被 job_id 丢弃）
        if self._dbg_worker and self._dbg_worker.is_alive():
            self._dbg_cancel.set()
        self._dbg_cancel = threading.Event()

        self._last_det = None

        def _worker():
            ok = False; det = None
            try:
                if bool(self.use_pipeline_algo.get()):
                    try:
                        dn = bool(self.app.enable_strong_denoise.get())
                        min_r = int(self.min_r.get()); max_r = int(self.max_r.get())
                        p1 = int(self.param1.get()); p2 = int(self.param2.get())
                        circle, processed, quality, method, brightness = _algo_circle.detect_circle_phd2_enhanced(
                            img, min_r, max_r, p1, p2, strong_denoise=dn
                        )
                        if circle is not None:
                            det = {"circle": (float(circle[0]), float(circle[1]), float(circle[2])),
                                   "quality": float(quality),
                                   "proc_shape": img.shape[:2],
                                   "used_fallback": False}
                            ok = True
                        else:
                            det = None
                            ok = False
                    except Exception:
                        det = None
                        ok = False
                else:
                    det, _used_fb = self._detect_best(img)
                    if det is not None:
                        det['used_fallback'] = det.get('used_fallback', False) or bool(_used_fb)
                        ok = True
            except Exception as e:
                det = None
            finally:
                self._dbg_queue.put((job_id, ok, det))

        self._dbg_worker = threading.Thread(target=_worker, daemon=True)
        self._dbg_worker.start()

    def _poll_debug_results(self):
        """定时从队列取结果；丢弃过期任务；必要时触发下一次计算。"""
        try:
            while True:
                job_id, ok, det = self._dbg_queue.get_nowait()
                if job_id != self._dbg_job_id:
                    continue  # 过期
                self._dbg_busy = False
                # 更新最近一次结果并重绘
                self._last_det = det if ok else None
                self.refresh()
        except queue.Empty:
            pass

        # 若期间发生了参数变化，且现在空闲，则再开一次
        if (not self._dbg_busy) and self._dbg_pending and self.preview_img_cv is not None and bool(self.enable_debug.get()):
            self._start_debug_compute()

        # 继续轮询
        self.after(40, self._poll_debug_results)

try:
    from ttkthemes import ThemedTk
except Exception:
    ThemedTk = None

try:
    from scipy.fft import fft2  # 探测
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

class PreviewWindow(tk.Toplevel):
    def __init__(self, master, app_controller):
        super().__init__(master)
        self.app = app_controller
        self.title("预览与半径估计")
        self.geometry("1100x650"); self.minsize(900, 500)
        self.center_window(); self.configure_fonts()
        self.preview_img_cv = None; self.preview_img_disp = None
        self.current_preview_path = None; self.preview_scale = 1.0
        self.rect_state = {"start": None, "rect": None}
        self.delta_var = tk.IntVar(value=100)
        self.estimate_radius_px = tk.IntVar(value=0)
        # 新增：自动检测相关变量
        self.delta_pct_var = tk.DoubleVar(value=3.0)
        self.detected_circle = None  # (cx, cy, r) in image pixels
        self.estimate_center_xy = None  # 新增：框选矩形中心（图像坐标）
        # 预览窗中的强力降噪勾选，默认跟随主界面
        try:
            self.strong_denoise_var = tk.BooleanVar(value=bool(self.app.enable_strong_denoise.get()))
        except Exception:
            self.strong_denoise_var = tk.BooleanVar(value=False)
        self._create_widgets(); self._bind_events(); self._show_initial_hint()

    def center_window(self):
        self.update_idletasks()
        w,h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth()//2) - (w//2)
        y = (self.winfo_screenheight()//2) - (h//2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def configure_fonts(self):
        try:
            self.default_font = DEFAULT_FONT
        except Exception:
            self.default_font = ("TkDefaultFont", 9)

    def _create_widgets(self):
        main = ttk.Frame(self, padding=10); main.pack(fill="both", expand=True)
        main.rowconfigure(2, weight=1); main.columnconfigure(0, weight=1)  # 修改为第2行是画布
        
        # 第一行工具条
        tool1 = ttk.Frame(main); tool1.grid(row=0, column=0, sticky="ew", pady=(0,5))
        ttk.Button(tool1, text="选择样张", command=self.choose_preview_image).pack(side="left", padx=(0,10))
        ttk.Label(tool1, text="增减范围 Δ:").pack(side="left", padx=(10,5))
        ttk.Spinbox(tool1, from_=0, to=5000, textvariable=self.delta_var, width=8).pack(side="left")
        self.est_label = ttk.Label(tool1, text=" | 估计半径: —"); self.est_label.pack(side="left", padx=(10,5))
        
        # 第二行工具条
        tool2 = ttk.Frame(main); tool2.grid(row=1, column=0, sticky="ew", pady=(0,10))
        ttk.Button(tool2, text="检测半径", command=self.detect_radius).pack(side="left", padx=(0,6))
        ttk.Label(tool2, text="Δ%:").pack(side="left", padx=(0,4))
        ttk.Spinbox(tool2, from_=0.5, to=10.0, increment=0.5, textvariable=self.delta_pct_var, width=5).pack(side="left")
        ttk.Button(tool2, text="应用检测半径和参考图像", command=self.apply_detected_radius).pack(side="left", padx=(8,0))
        ttk.Checkbutton(tool2, text="强力降噪(仅检测)", variable=self.strong_denoise_var).pack(side="left", padx=(14,0))
        
        # 画布
        self.canvas = tk.Canvas(main, background="#333", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")

    def _bind_events(self):
        self.canvas.bind("<Configure>", self._render_preview)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

    def _show_initial_hint(self):
        self.canvas.delete("hint"); self.canvas.update_idletasks()
        cw,ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            self.after(100, self._show_initial_hint); return
        font_size = 20 if (SYSTEM=="Darwin") else 16
        self.canvas.create_text(cw/2, ch/2, text="请选择样张，在图上拖拽鼠标框选月亮",
                                font=(self.default_font[0], font_size),
                                fill="gray60", tags="hint")

    def choose_preview_image(self):
        initdir = self.app.input_var.get() if os.path.isdir(self.app.input_var.get()) else os.getcwd()
        filetypes = [("支持的图像", " ".join(f"*{ext}" for ext in SUPPORTED_EXTS)), ("所有文件", "*.*")]
        path = filedialog.askopenfilename(title="选择样张用于预览与框选", filetypes=filetypes, initialdir=initdir, parent=self)
        if not path: return
        path = normalize_path(path); img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("错误", "无法读取该图像。", parent=self); return
        self.preview_img_bgr = img  # 保留BGR给检测
        self.preview_img_cv = to_display_rgb(img)  # RGB用于显示
        self.detected_circle = None  # 重置检测结果
        self.estimate_center_xy = None
        self.current_preview_path = path
        self.title(f"预览与半径估计 - {os.path.basename(path)}")
        self._render_preview()

    def _render_preview(self, event=None):
        self.canvas.delete("all")
        if self.preview_img_cv is None:
            self._show_initial_hint(); return
        h,w = self.preview_img_cv.shape[:2]; cw,ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.preview_scale = min(cw/w, ch/h, 1.0)
        nw, nh = int(w*self.preview_scale), int(h*self.preview_scale)
        try:
            disp = Image.fromarray(self.preview_img_cv).resize((nw, nh), Image.LANCZOS)
            self.preview_img_disp = ImageTk.PhotoImage(disp)
            self.canvas.create_image(cw/2, ch/2, image=self.preview_img_disp, anchor="center")
        except Exception as e:
            print(f"图像显示失败: {e}")
        # 若已有检测结果，则叠加红色圆与圆心
        if self.detected_circle is not None:
            try:
                cx, cy, r = self.detected_circle
                x, y = self._img_to_canvas(cx, cy)
                s = self.preview_scale
                self.canvas.create_oval(x - r*s, y - r*s, x + r*s, y + r*s, outline="#FF4D4F", width=2, tags="det")
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#FF4D4F", outline="", tags="det")
            except Exception:
                pass
        self.rect_state = {"start": None, "rect": None}
        self.estimate_radius_px.set(0); self.est_label.config(text=" | 估计半径: —")

    def _img_to_canvas(self, xi, yi):
        """把图像坐标映射到画布坐标（与 _to_image_coords 相反）。"""
        if self.preview_img_cv is None:
            return 0, 0
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        h, w = self.preview_img_cv.shape[:2]
        nw, nh = int(w*self.preview_scale), int(h*self.preview_scale)
        ox, oy = (cw-nw)//2, (ch-nh)//2
        xc = ox + xi * self.preview_scale
        yc = oy + yi * self.preview_scale
        return xc, yc

    def detect_radius(self):
        """在当前预览图上自动检测月面圆并叠加显示。"""
        try:
            import numpy as _np
            import cv2 as _cv2
        except Exception:
            messagebox.showerror("错误", "缺少依赖 numpy/cv2，无法检测。", parent=self)
            return
        if getattr(self, 'preview_img_bgr', None) is None:
            messagebox.showwarning("提示", "请先选择一张样张。", parent=self)
            return
        # require a manual estimate (drag box) and use Δ in pixels to constrain Hough
        r_est = int(self.estimate_radius_px.get())
        if r_est <= 0:
            messagebox.showwarning("提示", "请先在图像上拖出一个方框以估计半径，然后再点击'检测半径'。", parent=self)
            return
        dpx = max(1, int(self.delta_var.get()))
        min_r = max(1, r_est - dpx)
        max_r = max(min_r + 1, r_est + dpx)
        # 选一个相对快速的缩略图尺度（与主流程一致）
        img = self.preview_img_bgr
        H0, W0 = img.shape[:2]
        MAX_SIDE = 1600
        s = max(H0, W0) / float(MAX_SIDE)
        if s > 1:
            Hs = int(round(H0 / s)); Ws = int(round(W0 / s))
            small = _cv2.resize(img, (Ws, Hs), interpolation=_cv2.INTER_AREA)
        else:
            s = 1.0; Hs, Ws = H0, W0; small = img.copy()

        # 使用"估计半径 ± Δ(像素)"作为检测范围
        p1 = int(self.app.params["param1"].get())
        p2 = int(self.app.params["param2"].get())
        min_r_s = max(1, int(round(min_r / s)))
        max_r_s = max(min_r_s + 1, int(round(max_r / s)))

        # 期望圆心（来自框选矩形中心），用于约束/打分
        exp_cxcy = self.estimate_center_xy
        center_tol_abs = max(12.0, 0.6 * float(r_est))  # 允许的最大偏移（原图像素）

        # 优先走 algorithms_circle 提供的调试接口
        det = None
        try:
            try:
                if exp_cxcy is not None:
                    det = _algo_circle.debug_detect_circle(
                        small, min_r_s, max_r_s, p1, p2,
                        expected_center=(float(exp_cxcy[0])/s, float(exp_cxcy[1])/s),
                        center_dist_max=(center_tol_abs / s)
                    )
                else:
                    det = _algo_circle.debug_detect_circle(small, min_r_s, max_r_s, p1, p2)
            except TypeError:
                # 兼容旧签名
                det = _algo_circle.debug_detect_circle(small, min_radius=min_r_s, max_radius=max_r_s, param1=p1, param2=p2)
        except Exception:
            det = None

        # 若失败，退回一次轻量 Hough 尝试
        if not det or det.get('circle') is None:
            try:
                gray = _cv2.cvtColor(small, _cv2.COLOR_BGR2GRAY) if small.ndim==3 else small
                # HoughCircles 需要 8-bit 图；对 16-bit/浮点做拉伸到 0..255
                if gray.dtype != _np.uint8:
                    g32 = gray.astype(_np.float32)
                    p1, p2p = _np.percentile(g32, [1.0, 99.7])
                    if p2p <= p1 + 1e-6:
                        gray8 = _cv2.normalize(g32, None, 0, 255, _cv2.NORM_MINMAX).astype(_np.uint8)
                    else:
                        norm = _np.clip((g32 - p1) / (p2p - p1), 0, 1) * 255.0
                        gray8 = norm.astype(_np.uint8)
                else:
                    gray8 = gray
                # 预览中的强力降噪（仅用于检测）
                try:
                    if bool(self.strong_denoise_var.get()):
                        gray8 = _cv2.fastNlMeansDenoising(gray8, None, h=10, templateWindowSize=7, searchWindowSize=21)
                        gray8 = _cv2.medianBlur(gray8, 3)
                except Exception:
                    pass
                gray8 = _cv2.GaussianBlur(gray8, (3,3), 0)
                minDist = max(30, min(gray8.shape[:2])//4)
                circles = _cv2.HoughCircles(gray8, _cv2.HOUGH_GRADIENT, dp=1.2, minDist=minDist,
                                             param1=max(30,p1), param2=max(12,p2),
                                             minRadius=min_r_s, maxRadius=max_r_s)
                if circles is not None:
                    circles = _np.squeeze(circles, axis=0)
                    # 若有中心先验，优先在阈值内挑选"就近且质量高"的圆
                    if exp_cxcy is not None:
                        try:
                            if hasattr(_algo_circle, 'pick_best_circle'):
                                best = _algo_circle.pick_best_circle(
                                    gray8,
                                    circles,
                                    expected_center=(float(exp_cxcy[0])/s, float(exp_cxcy[1])/s),
                                    center_weight=0.04
                                )
                            else:
                                best = None
                        except Exception:
                            best = None
                        if best is None:
                            ex, ey = float(exp_cxcy[0])/s, float(exp_cxcy[1])/s
                            tol_s = center_tol_abs / s
                            def _ok(c):
                                return (_np.hypot(c[0]-ex, c[1]-ey) <= tol_s)
                            cand = [c for c in circles if _ok(c)] or list(circles)
                            best = max(cand, key=lambda c: c[2])
                    else:
                        best = max(circles, key=lambda c: c[2])
                    x, y, r = best
                    det = {"circle": (float(x), float(y), float(r))}
            except Exception:
                det = None

        if not det or det.get('circle') is None:
            messagebox.showwarning("提示", "未检测到圆。\n请适当调整'增减范围 Δ'或重新框选月亮使估计半径更接近，然后重试。", parent=self)
            return

        cx, cy, r = det['circle']
        # 还原到原图尺度
        cx, cy, r = cx * s, cy * s, r * s

        # 圆心与框选中心的距离约束：若偏差过大则提醒并忽略此次结果
        if exp_cxcy is not None:
            d = ((cx - exp_cxcy[0])**2 + (cy - exp_cxcy[1])**2) ** 0.5
            if d > center_tol_abs:
                messagebox.showwarning(
                    "提示",
                    "检测到的圆心与框选区域中心偏差较大，已忽略本次结果。\n"
                    "请尝试更精准地框选月面，或调整 Δ 后重试。",
                    parent=self
                )
                return

        self.detected_circle = (float(cx), float(cy), float(r))
        # 立即重绘叠加
        self._render_preview()
        # 同时在标签处显示检测到的半径和范围
        try:
            self.est_label.config(text=f" | 估计半径: {int(round(r))} px  (自动检测，范围 {min_r}–{max_r})")
        except Exception:
            pass

    def apply_detected_radius(self):
        """将自动检测到的半径按 Δ% 反馈到主界面 min/max 半径。"""
        if self.detected_circle is None:
            messagebox.showwarning("提示", "请先点击『检测半径』并确保检测成功。", parent=self)
            return
        r = float(self.detected_circle[2])
        pct = max(0.1, float(self.delta_pct_var.get())) / 100.0
        min_r = max(1, int(round(r * (1.0 - pct))))
        max_r = max(min_r + 1, int(round(r * (1.0 + pct))))
        self.app.params["min_radius"].set(min_r)
        self.app.params["max_radius"].set(max_r)
        # 同步强力降噪开关到主界面
        try:
            self.app.enable_strong_denoise.set(bool(self.strong_denoise_var.get()))
        except Exception:
            pass
        # 同时将当前预览图作为参考图像，便于后续批量对齐
        if getattr(self, 'current_preview_path', None):
            self.app.reference_image_var.set(self.current_preview_path)
            ref_note = f"\n参考图像: {os.path.basename(self.current_preview_path)}"
        else:
            ref_note = ""
        messagebox.showinfo(
            "已应用",
            f"检测半径 r≈{int(round(r))} px\nΔ%={pct*100:.1f}%\n\n已设置范围:\nMin={min_r}\nMax={max_r}{ref_note}",
            parent=self
        )

    def _to_image_coords(self, xc, yc):
        if self.preview_img_cv is None: return 0,0
        cw,ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        h,w = self.preview_img_cv.shape[:2]
        nw,nh = int(w*self.preview_scale), int(h*self.preview_scale)
        ox,oy = (cw-nw)//2, (ch-nh)//2
        xi = int((xc-ox)/self.preview_scale); yi = int((yc-oy)/self.preview_scale)
        return max(0, min(w-1, xi)), max(0, min(h-1, yi))

    def on_canvas_press(self, e):
        self.canvas.delete("hint")
        if self.preview_img_cv is None: return
        if self.rect_state["rect"]:
            self.canvas.delete(self.rect_state["rect"]); self.rect_state["rect"] = None
        self.rect_state["start"] = (e.x, e.y)

    def on_canvas_drag(self, e):
        if self.rect_state["start"] is None: return
        x0, y0 = self.rect_state["start"]
        if self.rect_state["rect"] is None:
            self.rect_state["rect"] = self.canvas.create_rectangle(x0, y0, e.x, e.y, outline="#00BFFF", width=2)
        else:
            self.canvas.coords(self.rect_state["rect"], x0, y0, e.x, e.y)

    def on_canvas_release(self, e):
        if self.rect_state["start"] is None: return
        x0, y0 = self.rect_state["start"]
        xi0, yi0 = self._to_image_coords(x0, y0)
        xi1, yi1 = self._to_image_coords(e.x, e.y)
        w_px, h_px = abs(xi1 - xi0), abs(yi1 - yi0)
        self.rect_state["start"] = None
        if w_px < 4 or h_px < 4:
            if self.rect_state["rect"]:
                self.canvas.delete(self.rect_state["rect"]); self.rect_state["rect"] = None
            return
        radius = int(min(w_px, h_px)/2)
        # 保存框选中心（图像坐标），用于后续圆心距离约束
        cx_est = (xi0 + xi1) / 2.0
        cy_est = (yi0 + yi1) / 2.0
        self.estimate_center_xy = (float(cx_est), float(cy_est))
        self.estimate_radius_px.set(radius)
        self.est_label.config(text=f" | 估计半径: {radius} px")

    def apply_to_main_sliders(self):
        r = self.estimate_radius_px.get()
        if r <= 0:
            messagebox.showwarning("提示", "请先在图像上框选一个月球区域来估计半径。", parent=self); return
        d = max(0, self.delta_var.get())
        min_r = max(1, r-d); max_r = max(min_r+1, r+d)
        self.app.params["min_radius"].set(min_r)
        self.app.params["max_radius"].set(max_r)
        if getattr(self, 'current_preview_path', None):
            use_as_ref = messagebox.askyesno("设置参考图像",
                        f"是否将当前预览的图像设为参考图像？\n\n图像: {os.path.basename(self.current_preview_path)}\n估计半径: {r} px", parent=self)
            if use_as_ref:
                self.app.reference_image_var.set(self.current_preview_path)
        messagebox.showinfo("成功", f"半径范围已成功更新为:\nMin: {min_r}\nMax: {max_r}", parent=self)

class ProgressWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("处理进度"); self.geometry("400x150"); self.resizable(False, False)
        self.transient(master); self.grab_set()
        main = ttk.Frame(self, padding=20); main.pack(fill="both", expand=True)
        self.status_label = ttk.Label(main, text="准备开始...", font=UI_FONT); self.status_label.pack(pady=(0,10))
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(main, variable=self.progress_var, maximum=100); self.progress_bar.pack(fill="x", pady=(0,10))
        self.percent_label = ttk.Label(main, text="0%", font=UI_FONT); self.percent_label.pack()
        self.center_window()
    def center_window(self):
        self.update_idletasks()
        w,h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth()//2) - (w//2)
        y = (self.winfo_screenheight()//2) - (h//2)
        self.geometry(f"{w}x{h}+{x}+{y}")
    def update_progress(self, progress, status):
        self.progress_var.set(progress)
        self.status_label.config(text=status)
        self.percent_label.config(text=f"{progress}%")
        self.update()

class UniversalLunarAlignApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"月食圆面对齐工具 V{VERSION} By @正七价的氟离子")
        self.root.geometry("920x800"); self.root.minsize(750, 700)
        self.setup_cross_platform()
        self.preview_window = None; self.progress_window = None
        self.debug_window = None  # 独立调试窗口实例引用
        self._about_photo = None  # cache to avoid GC in About window
        self._qr_photo = None  # cache QR image to avoid GC
        self._init_vars()
        self._create_main_layout(); self._create_path_widgets()
        self._create_param_widgets(); self._create_imppg_widgets()
        self._create_debug_widgets(); self._create_action_widgets()
        self._create_log_widgets(); self._set_initial_log_message()
        self.on_debug_mode_change(); self.on_advanced_change()

    def setup_cross_platform(self):
        try:
            if SYSTEM == "Windows":
                self.root.iconbitmap(default='')
        except Exception:
            pass
        style = ttk.Style()
        try:
            style.theme_use('winnative' if SYSTEM=="Windows" else 'aqua' if SYSTEM=="Darwin" else 'clam')
        except Exception:
            pass

    def _init_vars(self):
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.reference_image_var = tk.StringVar()
        self.debug_var = tk.BooleanVar(value=DEFAULT_DEBUG_MODE)
        self.debug_image_path_var = tk.StringVar(value="")
        self.params = {
            "min_radius": tk.IntVar(value=300),
            "max_radius": tk.IntVar(value=800),
            "param1": tk.IntVar(value=50),
            "param2": tk.IntVar(value=30)
        }
        self.use_advanced_alignment = tk.BooleanVar(value=False)
        self.alignment_method = tk.StringVar(value="auto")
        # 新增：强力降噪（仅用于检测/对齐，不影响输出）
        self.enable_strong_denoise = tk.BooleanVar(value=False)

    def _create_main_layout(self):
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(2, weight=1)
        control = ttk.Frame(self.root, padding=(10,10,10,0))
        control.grid(row=0, column=0, sticky="ew"); control.columnconfigure(0, weight=1); control.columnconfigure(1, weight=0)
        self.path_frame = ttk.LabelFrame(control, text="1. 路径设置", padding=10)
        self.path_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,5))
        params_container = ttk.Frame(control); params_container.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        params_container.columnconfigure(0, weight=2); params_container.columnconfigure(1, weight=1)
        self.param_frame = ttk.LabelFrame(params_container, text="2. PHD2霍夫圆参数", padding=10)
        self.param_frame.grid(row=0, column=0, sticky="nsew", padx=(0,5))
        # 文案替换：IMPPG -> 多ROI精配准（保持控件结构不变）
        self.imppg_frame = ttk.LabelFrame(params_container, text="3. 多ROI精配准", padding=10)
        self.imppg_frame.grid(row=0, column=1, sticky="nsew", padx=(5,0))
        self.debug_frame = ttk.LabelFrame(control, text="4. 预览与调试", padding=10)
        self.debug_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        self.action_frame = ttk.Frame(self.root, padding=(0,10))
        self.action_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _create_path_widgets(self):
        f = self.path_frame; f.columnconfigure(1, weight=1)
        ttk.Label(f, text="输入文件夹:", font=UI_FONT).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(f, textvariable=self.input_var, font=UI_FONT).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(f, text="浏览...", command=self.select_input_folder).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(f, text="输出文件夹:", font=UI_FONT).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(f, textvariable=self.output_var, font=UI_FONT).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(f, text="浏览...", command=self.select_output_folder).grid(row=1, column=2, padx=5, pady=5)
        ttk.Label(f, text="参考图像:", font=UI_FONT).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.reference_entry = ttk.Entry(f, textvariable=self.reference_image_var, font=UI_FONT)
        self.reference_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        fb = ttk.Frame(f); fb.grid(row=2, column=2, padx=5, pady=5)
        ttk.Button(fb, text="选择", command=self.select_reference_image).pack(side="left")
        ttk.Button(fb, text="清除", command=self.clear_reference_image).pack(side="left", padx=(2,0))
        
        # 修改提示行，把强力降噪勾选框放到提示右边
        help_row = ttk.Frame(f)
        help_row.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=(2,5))
        help_row.columnconfigure(0, weight=1)
        
        help_text = ttk.Label(help_row, text="💡参考图像：作为对齐基准的图像。请在预览&半径估计窗口选择。",
                              font=(UI_FONT[0], UI_FONT[1]-1), foreground="gray")
        help_text.grid(row=0, column=0, sticky="w")
        
        ttk.Checkbutton(help_row, text="强力降噪(仅检测/对齐)", variable=self.enable_strong_denoise).grid(row=0, column=1, sticky="e", padx=(10,0))

    def _create_param_widgets(self):
        f = self.param_frame; f.columnconfigure(1, weight=1)
        help_text = ("• PHD2增强算法：三级检测策略，自适应图像亮度\n"
                     "• 最小/最大半径: 限制检测到的圆的半径范围(像素)\n"
                     "• 参数1: Canny边缘检测高阈值\n"
                     "• 参数2: 霍夫累加器阈值（关键参数）")
        ttk.Label(f, text=help_text, justify="left", font=(UI_FONT[0], UI_FONT[1]-1)).grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=(0,10))
        defs = [("min_radius","最小半径:",1,3000),("max_radius","最大半径:",10,4000),("param1","参数1 (Canny):",1,200),("param2","参数2 (累加阈值):",1,100)]
        for i,(k,label,a,b) in enumerate(defs):
            var = self.params[k]; r = i+1
            ttk.Label(f, text=label, font=UI_FONT).grid(row=r, column=0, sticky="w", padx=5, pady=3)
            ttk.Scale(f, from_=a, to_=b, orient="horizontal", variable=var,
                      command=lambda v, kk=k: self.params[kk].set(int(float(v)))).grid(row=r, column=1, sticky="ew", padx=5, pady=3)
            ttk.Spinbox(f, from_=a, to_=b, textvariable=var, width=6, font=UI_FONT).grid(row=r, column=2, padx=5, pady=3)

    def _create_imppg_widgets(self):
        f = self.imppg_frame
        # 文案替换：启用IMPPG算法 -> 启用多ROI精配准
        ttk.Checkbutton(f, text="启用多ROI精配准(仅支持赤道仪跟踪拍摄的素材)", variable=self.use_advanced_alignment,
                        command=self.on_advanced_change).pack(fill="x", padx=5, pady=(0,10))
        # 文案替换：算法类型 -> 算法说明（保持 combobox，不影响 pipeline）
        ttk.Label(f, text="算法说明:", font=UI_FONT).pack(anchor="w", padx=5)
        self.method_combo = ttk.Combobox(f, textvariable=self.alignment_method,
                                         values=['auto','phase_corr','template','feature','centroid'],
                                         state="disabled", width=15, font=UI_FONT)
        self.method_combo.pack(fill="x", padx=5, pady=2)
        # 文案替换：算法帮助
        algo_help = ("• 在月盘内自动选择多块ROI进行 ZNCC/相位相关微调\n"
                     "• 对亮度变化与阴影边界更鲁棒，失败时自动回退到圆心对齐\n"
                     "• 建议在偏食/生光阶段启用，多数情况默认关闭即可")
        ttk.Label(f, text=algo_help, justify="left",
                  font=(UI_FONT[0], UI_FONT[1]-2), foreground="darkgreen").pack(anchor="w", padx=5, pady=(5,10))
        ttk.Label(f, text="⚠️ 实验性功能，不推荐开启", font=(UI_FONT[0], UI_FONT[1]-1),
                  foreground="orange", justify="center").pack(pady=5)

    def _create_debug_widgets(self):
        f = self.debug_frame; f.columnconfigure(0, weight=1); f.columnconfigure(1, weight=1)
        # 修改为左右各一半的布局
        ttk.Button(f, text="打开预览 & 半径估计窗口", command=self.open_preview)\
            .grid(row=0, column=0, sticky="ew", padx=(5,2), pady=5)
        # 新增：打开独立调试窗口
        ttk.Button(f, text="打开调试窗口（实时参数预览）", command=self.open_debug)\
            .grid(row=0, column=1, sticky="ew", padx=(2,5), pady=5)

    def on_debug_mode_change(self):
        # 独立调试窗口模式下，这里可能没有旧的条目控件，做容错即可
        en = bool(self.debug_var.get())
        entry = getattr(self, 'debug_entry', None)
        btn = getattr(self, 'debug_button', None)
        state = "normal" if en else "disabled"
        if entry is not None:
            entry.config(state=state)
        if btn is not None:
            btn.config(state=state)
        if not en:
            self.debug_image_path_var.set("")
    def open_debug(self):
        # 复用已有调试窗口；若不存在则创建一个
        if getattr(self, "debug_window", None) and self.debug_window.winfo_exists():
            self.debug_window.deiconify()
            self.debug_window.lift()
            self.debug_window.focus_force()
            return

        self.debug_window = DebugWindow(self.root, self)

        def _on_close():
            try:
                self.debug_window.destroy()
            except Exception:
                pass
            finally:
                self.debug_window = None

        # 关闭时清理引用，避免再次打开不生效
        self.debug_window.protocol("WM_DELETE_WINDOW", _on_close)
        self.debug_window.lift()
        self.debug_window.focus_force()

    def on_advanced_change(self):
        self.method_combo.config(state="readonly" if self.use_advanced_alignment.get() else "disabled")

    def _create_action_widgets(self):
        f = self.action_frame; f.columnconfigure(0, weight=1)
        
        # 修改为水平布局：开始对齐按钮在左，打赏作者按钮在右
        action_row = ttk.Frame(f)
        action_row.pack(fill="x", pady=10, padx=200)
        action_row.columnconfigure(0, weight=1)
        
        self.start_button = ttk.Button(action_row, text="🚀 开始集成对齐", command=self.start_alignment)
        self.start_button.grid(row=0, column=0, sticky="ew", ipady=8, padx=(0,10))
        
        try:
            style = ttk.Style()
            style.configure("Accent.TButton", font=(UI_FONT[0], UI_FONT[1]+1, "bold"))
            self.start_button.configure(style="Accent.TButton")
        except Exception:
            pass
        
        # 打赏作者按钮放在右边
        ttk.Button(action_row, text="打赏作者", command=self.show_about_author).grid(row=0, column=1, sticky="e")

    def _create_log_widgets(self):
        lp = ttk.Frame(self.root, padding=(10,5,10,10))
        lp.grid(row=2, column=0, columnspan=2, sticky="nsew")
        lp.columnconfigure(0, weight=1); lp.rowconfigure(0, weight=1)
        self.log_box = scrolledtext.ScrolledText(lp, height=12, wrap="word", relief="solid", borderwidth=1, font=UI_FONT)
        self.log_box.pack(fill="both", expand=True)

    def _set_initial_log_message(self):
        scipy_status = "✓ 已安装" if SCIPY_AVAILABLE else "✗ 未安装"
        welcome = (f"欢迎使用月食圆面对齐工具 V{VERSION} - 集成版 By @正七价的氟离子\n"
                   f"运行平台: {SYSTEM}\n"
                   f"SciPy状态: {scipy_status}\n"
                   "================================================================\n\n"
                   "算法说明：\n"
                   "• PHD2增强算法：基于霍夫圆检测，适用于完整清晰的月球\n"
                   "• 多ROI精配准：适用于偏食、生光等复杂阶段（实验性）\n"
                   "• 回退机制：确保在任何情况下都有可用的对齐方案\n\n"
                   "使用建议：\n"
                   "• 使用预览工具准确估算半径范围\n"
                   "• 参数2（累加器阈值）是最关键的调整参数\n"
                   f"• 支持格式：{', '.join(SUPPORTED_EXTS)}\n")
        if not SCIPY_AVAILABLE:
            welcome += ("\n⚠️ 注意: SciPy未安装，相位相关算法将被禁用\n"
                        "可通过 pip install scipy 安装以启用多ROI中的相位相关增强\n")
        self.log_box.insert(tk.END, welcome); self.log_box.config(state="disabled")

    # —— 选择/打开等 UI 行为（保持原文案） ——
    def select_input_folder(self):
        path = filedialog.askdirectory(title="选择输入文件夹")
        if path:
            path = normalize_path(path); self.input_var.set(path)
            parent = os.path.dirname(path); name = os.path.basename(path)
            self.output_var.set(safe_join(parent, f"{name}_aligned_v12b"))

    def select_output_folder(self):
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path: self.output_var.set(normalize_path(path))

    def select_reference_image(self):
        initdir = self.input_var.get() if os.path.isdir(self.input_var.get()) else os.getcwd()
        filetypes = [("支持的图像"," ".join(f"*{ext}" for ext in SUPPORTED_EXTS)), ("所有文件","*.*")]
        path = filedialog.askopenfilename(title="选择参考图像（用作对齐基准）", filetypes=filetypes, initialdir=initdir)
        if path:
            path = normalize_path(path)
            input_folder = self.input_var.get().strip()
            if input_folder and not path.startswith(input_folder):
                ok = messagebox.askyesno("确认",
                    "选择的参考图像不在输入文件夹内。\n建议选择输入文件夹中的图像作为参考。\n是否继续使用此图像？",
                    icon='question')
                if not ok: return
            self.reference_image_var.set(path)

    def clear_reference_image(self):
        self.reference_image_var.set("")

    def select_debug_image(self):
        initdir = self.input_var.get() if os.path.isdir(self.input_var.get()) else os.getcwd()
        filetypes = [("支持的图像"," ".join(f"*{ext}" for ext in SUPPORTED_EXTS)), ("所有文件","*.*")]
        path = filedialog.askopenfilename(title="选择调试样张", filetypes=filetypes, initialdir=initdir)
        if path: self.debug_image_path_var.set(normalize_path(path))

    def open_preview(self):
        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.preview_window = PreviewWindow(self.root, self)
        self.preview_window.deiconify(); self.preview_window.lift(); self.preview_window.focus_force()

    def show_progress_window(self):
        if self.progress_window is None or not self.progress_window.winfo_exists():
            self.progress_window = ProgressWindow(self.root)
        return self.progress_window

    def start_alignment(self):
        in_path = self.input_var.get().strip()
        out_path = self.output_var.get().strip()
        if not os.path.isdir(in_path):
            messagebox.showerror("错误", "请选择有效的输入文件夹。"); return
        if not out_path:
            messagebox.showerror("错误", "请指定输出文件夹。"); return

        use_advanced = self.use_advanced_alignment.get()
        method = self.alignment_method.get()
        if use_advanced and not SCIPY_AVAILABLE and method in ['auto','phase_corr']:
            ok = messagebox.askyesno("警告","SciPy未安装，相位相关算法将被禁用。\n多ROI精配准的相位相关增强可能受限。\n\n是否继续？", icon='warning')
            if not ok: return

        ref_path = self.reference_image_var.get().strip() or None
        ref_path = normalize_path(ref_path) if ref_path else None
        if ref_path and not os.path.exists(ref_path):
            ok = messagebox.askyesno("警告", f"指定的参考图像不存在：\n{ref_path}\n\n是否继续（将自动选择参考图像）？", icon='warning')
            if not ok: return
            ref_path = None

        dbg_mode = self.debug_var.get()
        dbg_path = self.debug_image_path_var.get().strip()
        dbg_basename = os.path.basename(dbg_path) if dbg_path else ""
        if dbg_mode and not dbg_basename:
            ok = messagebox.askyesno("提示","已开启调试模式，但未选择调试样张。\n处理将继续，但不会生成调试图像。\n是否继续？", icon='warning')
            if not ok: return

        hough_params = (
            self.params["min_radius"].get(),
            self.params["max_radius"].get(),
            self.params["param1"].get(),
            self.params["param2"].get()
        )

        self.log_box.config(state="normal"); self.log_box.delete(1.0, tk.END)
        # 文案替换：按钮状态文本
        self.start_button.config(state="disabled",
            text=("集成对齐中 (多ROI + PHD2)..." if use_advanced else "PHD2对齐中..."))
        pw = self.show_progress_window()

        def progress_callback(pct, status):
            if pw and pw.winfo_exists(): pw.update_progress(pct, status)

        import threading
        threading.Thread(
            target=align_moon_images_incremental,
            args=(in_path, out_path, hough_params, self.log_box, dbg_mode, dbg_basename,
                  self.on_alignment_complete, progress_callback, ref_path,
                  use_advanced, method, bool(self.enable_strong_denoise.get())),
            daemon=True
        ).start()

    def on_alignment_complete(self, success, message):
        self.root.after(0, lambda: self._update_ui_on_complete(success, message))

    def _update_ui_on_complete(self, success, message):
        self.start_button.config(state="normal", text="🚀 开始集成对齐")
        self.log_box.config(state="disabled")
        if self.progress_window and self.progress_window.winfo_exists():
            self.progress_window.destroy(); self.progress_window = None
        if success: messagebox.showinfo("处理完成", message)
        else: messagebox.showerror("处理失败", "处理过程中发生错误，详情请查看日志。", detail=message)

    def show_about_author(self):
        """弹出关于作者窗口，显示头像、说明与支付宝二维码（若存在）。"""
        top = tk.Toplevel(self.root)
        top.title("关于作者")
        top.resizable(False, False)
        top.transient(self.root)
        top.grab_set()

        frm = ttk.Frame(top, padding=16)
        frm.pack(fill="both", expand=True)

        # 新布局：0=文本区(含标题+头像)，1=分隔线，2=二维码
        frm.grid_columnconfigure(0, weight=1)   # text (with inline avatar)
        frm.grid_columnconfigure(1, weight=0)   # separator
        frm.grid_columnconfigure(2, weight=0)   # QR

        # 路径准备
        base_dir = os.path.dirname(os.path.abspath(__file__))
        avatar_path = None
        for name in ("avatar.jpg", "avatar.png", "avatar.jpeg"):
            p = os.path.join(base_dir, name)
            if os.path.exists(p):
                avatar_path = p
                break

        qr_path = None
        for name in ("QRcode.jpg", "QRcode.png", "QRcode.jpeg"):
            p = os.path.join(base_dir, name)
            if os.path.exists(p):
                qr_path = p
                break

        # 标题+头像并排（头像在标题右侧），横向填满，头像贴近右侧分隔线
        header = ttk.Frame(frm)
        # 让标题行横向填满，使头像可贴近右侧分隔线
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        header.grid_columnconfigure(0, weight=1)  # 标题占满左侧
        header.grid_columnconfigure(1, weight=0)  # 头像靠右

        title_lbl = ttk.Label(header, text="正七价的氟离子", font=(UI_FONT[0], UI_FONT[1] + 4, "bold"))
        title_lbl.grid(row=0, column=0, sticky="w")

        # 头像放在标题右侧，并与右侧分隔线对齐（靠右）
        avatar_lbl = ttk.Label(header)
        avatar_lbl.grid(row=0, column=1, sticky="e", padx=(10, 0))
        try:
            if avatar_path:
                im = Image.open(avatar_path).convert("RGBA")
                max_side = 100  # 放大头像，并与右侧分隔线对齐更醒目
                scale = min(max_side / im.width, max_side / im.height, 1.0)
                if scale < 1.0:
                    im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
                self._about_photo = ImageTk.PhotoImage(im)
                avatar_lbl.configure(image=self._about_photo)
        except Exception:
            pass

        desc = (
            "在家带娃的奶妈，不会写程序的天文爱好者不是老司机。\n"
            "感谢使用《月食圆面对齐工具》，欢迎反馈与交流！\n"
            "如果您愿意，欢迎支持一点养娃的奶粉钱（右侧支付宝二维码）。"
        )
        ttk.Label(
            frm,
            text=desc,
            justify="left",
            wraplength=440,   # 控制换行，不会顶到二维码
        ).grid(row=1, column=0, sticky="nw")

        # 垂直分隔线
        sep = ttk.Separator(frm, orient="vertical")
        sep.grid(row=0, column=1, rowspan=3, sticky="ns", padx=14)

        # 右：二维码与说明
        qr_panel = ttk.Frame(frm)
        qr_panel.grid(row=0, column=2, rowspan=3, sticky="ne")

        qr_label = ttk.Label(qr_panel)
        qr_label.pack(side="top", anchor="ne")

        try:
            if qr_path:
                qr = Image.open(qr_path).convert("RGBA")
                target_w = 240  # 稍收窄，避免喧宾夺主
                scale = min(target_w / qr.width, 1.0)
                if scale < 1.0:
                    qr = qr.resize((int(qr.width * scale), int(qr.height * scale)), Image.LANCZOS)
                self._qr_photo = ImageTk.PhotoImage(qr)
                qr_label.configure(image=self._qr_photo)
        except Exception:
            pass

        ttk.Label(qr_panel, text="支付宝 · 打赏支持", foreground="gray40").pack(side="top", pady=(6, 0))

        # 底部按钮条（右对齐）
        btn_bar = ttk.Frame(frm)
        btn_bar.grid(row=3, column=0, columnspan=3, sticky="e", pady=(12, 0))
        ttk.Button(btn_bar, text="关闭", command=top.destroy).pack(side="right")
