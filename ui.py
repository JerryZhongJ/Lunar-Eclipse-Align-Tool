# ui.py
import os, tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
import cv2

from utils_common import (
    VERSION, SYSTEM, DEFAULT_DEBUG_MODE, SUPPORTED_EXTS, UI_FONT, DEFAULT_FONT,
    normalize_path, safe_join, imread_unicode, to_display_rgb,
    force_garbage_collection, log
)

from pipeline import align_moon_images_incremental

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
        self.geometry("900x600"); self.minsize(600, 400)
        self.center_window(); self.configure_fonts()
        self.preview_img_cv = None; self.preview_img_disp = None
        self.current_preview_path = None; self.preview_scale = 1.0
        self.rect_state = {"start": None, "rect": None}
        self.delta_var = tk.IntVar(value=100)
        self.estimate_radius_px = tk.IntVar(value=0)
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
        main.rowconfigure(1, weight=1); main.columnconfigure(0, weight=1)
        tool = ttk.Frame(main); tool.grid(row=0, column=0, sticky="ew", pady=(0,10))
        ttk.Button(tool, text="选择样张", command=self.choose_preview_image).pack(side="left", padx=(0,10))
        ttk.Label(tool, text="增减范围 Δ:").pack(side="left", padx=(10,5))
        ttk.Spinbox(tool, from_=0, to=5000, textvariable=self.delta_var, width=8).pack(side="left")
        self.est_label = ttk.Label(tool, text=" | 估计半径: —"); self.est_label.pack(side="left", padx=(10,5))
        ttk.Button(tool, text="✔ 应用到主界面", command=self.apply_to_main_sliders).pack(side="right")
        self.canvas = tk.Canvas(main, background="#333", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")

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
        self.preview_img_cv = to_display_rgb(img)
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
        self.rect_state = {"start": None, "rect": None}
        self.estimate_radius_px.set(0); self.est_label.config(text=" | 估计半径: —")

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
        help_text = ttk.Label(f, text="💡参考图像：作为对齐基准的图像。请在预览&半径估计窗口选择。",
                              font=(UI_FONT[0], UI_FONT[1]-1), foreground="gray")
        help_text.grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=(2,5))

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
        f = self.debug_frame; f.columnconfigure(1, weight=1)
        ttk.Button(f, text="打开预览 & 半径估计窗口", command=self.open_preview)\
            .grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=(0,10))
        ttk.Checkbutton(f, text="启用调试模式", variable=self.debug_var, command=self.on_debug_mode_change)\
            .grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.debug_entry = ttk.Entry(f, textvariable=self.debug_image_path_var, state="disabled", font=UI_FONT)
        self.debug_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.debug_button = ttk.Button(f, text="选择调试样张", command=self.select_debug_image, state="disabled")
        self.debug_button.grid(row=1, column=2, padx=5, pady=5)

    def on_debug_mode_change(self):
        en = self.debug_var.get()
        state = "normal" if en else "disabled"
        self.debug_entry.config(state=state); self.debug_button.config(state=state)
        if not en: self.debug_image_path_var.set("")

    def on_advanced_change(self):
        self.method_combo.config(state="readonly" if self.use_advanced_alignment.get() else "disabled")

    def _create_action_widgets(self):
        f = self.action_frame; f.columnconfigure(0, weight=1)
        self.start_button = ttk.Button(f, text="🚀 开始集成对齐", command=self.start_alignment)
        self.start_button.pack(pady=10, ipady=8, fill="x", padx=200)
        try:
            style = ttk.Style()
            style.configure("Accent.TButton", font=(UI_FONT[0], UI_FONT[1]+1, "bold"))
            self.start_button.configure(style="Accent.TButton")
        except Exception:
            pass
        # 关于作者按钮
        ttk.Button(f, text="关于作者", command=self.show_about_author).pack(pady=(0, 0))

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
            self.output_var.set(safe_join(parent, f"{name}_aligned_v11"))

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
                  use_advanced, method),
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
