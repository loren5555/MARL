import os
import logging
import time


class LoggerGenerator:

    def __init__(
            self,
            log_dir="./log",
            log_name=time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())) + ".log",
            logger_name=os.path.join("./log", time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())) + ".log"),
            console_log=True,
            console_level=logging.DEBUG,
            file_log=True,
            file_level=logging.DEBUG,
            fmt="{asctime} | {levelname:<8} | {message}",
            datefmt="%Y-%m-%d %H:%M:%S"
    ):
        log_dir = os.path.abspath(log_dir)
        if os.path.exists(log_dir) is False:
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, log_name)  # log文件名

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt, datefmt, style="{")

        if console_log is True:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            sh.setLevel(console_level)
            self.logger.addHandler(sh)

        if file_log is True:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(formatter)
            fh.setLevel(file_level)
            self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger

    # def debug(self, message):
    #     self.logger.debug(message)
    #
    # def info(self, message):
    #     self.logger.info(message)
    #
    # def warning(self, message):
    #     self.logger.warning(message)
    #
    # def error(self, message):
    #     self.logger.error(message)
    #
    # def critical(self, message):
    #     self.logger.critical(message)
