import logging
from lunar_eclipse_align.utils.constants import DEBUG
from PySide6.QtWidgets import QTextBrowser


class AnsiColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        no_style = "\033[0m"
        bold = "\033[91m"
        grey = "\033[90m"
        yellow = "\033[93m"
        red = "\033[31m"
        red_light = "\033[91m"
        start_style = {
            "DEBUG": grey,
            "INFO": no_style,
            "WARNING": yellow,
            "ERROR": red,
            "CRITICAL": red_light + bold,
        }.get(record.levelname, no_style)
        end_style = no_style
        return f"{start_style}{super().format(record)}{end_style}"


class HtmlFormatter(logging.Formatter):
    """HTML格式的彩色日志格式化器，用于QTextBrowser"""

    def format(self, record: logging.LogRecord):
        # 根据日志级别选择颜色
        color_map = {
            "DEBUG": "gray",
            "INFO": "black",
            "WARNING": "orange",
            "ERROR": "red",
            "CRITICAL": "darkred",
        }
        color = color_map.get(record.levelname, "black")

        # 格式化消息
        msg = super().format(record)

        # 包装为HTML
        # 对于CRITICAL级别，添加粗体
        if record.levelname in ["CRITICAL", "ERROR", "WARNING"]:
            return f'<span style="color:{color}; font-weight:bold">{msg}</span>'
        else:
            return f'<span style="color:{color}">{msg}</span>'


class TextBrowserHandler(logging.Handler):
    """A logging handler that outputs to a QTextBrowser widget."""

    def __init__(self, text_browser: QTextBrowser):
        super().__init__()
        self.text_browser = text_browser

    def emit(self, record: logging.LogRecord):
        scrollbar = self.text_browser.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= (scrollbar.maximum() - 5)

        msg = self.format(record)
        # Append the message to the QTextBrowser
        self.text_browser.append(msg)
        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())


def setup_logging():
    """设置日志记录器"""
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    if DEBUG:
        console_handler.setLevel(logging.DEBUG)
    console_formatter = AnsiColorFormatter("{levelname:<8s} | {message}", style="{")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # file_handler = logging.FileHandler('lunar_eclipse_align.log', mode='w', encoding='utf-8')
    # file_handler.setLevel(logging.DEBUG)
    # file_formatter = logging.Formatter(
    #     '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    # )
    # file_handler.setFormatter(file_formatter)
    # logger.addHandler(file_handler)


def enable_gui_logging(text_browser: QTextBrowser):
    """启用GUI日志记录"""
    gui_handler = TextBrowserHandler(text_browser)
    gui_handler.setLevel(logging.INFO)
    gui_formatter = HtmlFormatter("{levelname:<8s} | {message}", style="{")
    gui_handler.setFormatter(gui_formatter)
    logging.getLogger().addHandler(gui_handler)
