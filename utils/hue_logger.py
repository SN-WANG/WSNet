# A colorful logging utility with ANSI support
# Author: Shengning Wang

import sys
import logging

# Soft dependency for tqdm
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    tqdm = None


class HueLogger:
    """
    A colorful logging utility with ANSI escape code support.

    This class provides a pre-configured logger with color-coded output for 
    enhanced terminal readability. It ensures clean re-initialization by 
    clearing existing handlers, making it safe for interactive environments 
    like Jupyter notebooks where modules may be reloaded.

    Attributes:
        logger (logging.Logger): The configured logger instance.
    """

    # ANSI Color Codes
    b = "\033[1;34m"    # major key/parameter:      bold blue
    c = "\033[1;36m"    # minor key/parameter:      bold cyan
    m = "\033[1;35m"    # value/reading:            bold magenta
    y = "\033[1;33m"    # warning/highlighting:     bold yellow
    g = "\033[1;32m"    # success/save:             bold green
    r = "\033[1;31m"    # error/critical:           bold red

    q = "\033[0m"      # quit/reset

    def __init__(self, name: str = __name__, level: int = logging.INFO) -> None:
        """
        Initializes the SmartLogger with a tqdm-compatible stream handler.

        Args:
            name (str): The name of the logger, typically __name__.
            level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # disable propagation to prevent duplicate logs from the root logger
        self.logger.propagate = False

        # check for existing handlers to avoid redundant log outputs in singleton logger
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        handler: logging.StreamHandler = self._get_handler()
        self.logger.addHandler(handler)

    def _get_handler(self) -> logging.StreamHandler:
        """
        Constructs a StreamHandler with ANSI color support.
        Uses tqdm.write() if tqdm is available, otherwise standard emit.

        Returns:
            logging.StreamHandler: Configured handler with ANSI color support.
        """
        # define color-coded format for enhanced visibility in terminal, Green for timestamp, Blue for levelname
        log_format: str = f"\033[90m%(asctime)s{self.q} - {self.b}%(levelname)s{self.q} - %(message)s"
        formatter: logging.Formatter = logging.Formatter(log_format, "%H:%M:%S")

        # direct stream to stdout
        handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)

        # override emit to use tqdm.write, ensuring logs appear above progress bars
        if _HAS_TQDM:
            handler.emit = lambda record: tqdm.write(formatter.format(record))

        handler.setFormatter(formatter)
        return handler

hue = HueLogger()
logger: logging.Logger = hue.logger
