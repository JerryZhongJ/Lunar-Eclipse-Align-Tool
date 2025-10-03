import logging
from lunar_eclipse_align.utils.constants import DEBUG


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
