import os
import logging
import time


class MARLLogger:
    def __init__(
            self,
            log_dir="./log",
            log_name=time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())) + ".log",
            logger_name=None,
            console_log=True,
            console_level=logging.DEBUG,
            file_log=True,
            file_level=logging.DEBUG,
            fmt="{asctime} | {levelname:<8} | {message}",
            datefmt="%Y-%m-%d %H:%M:%S",
            propagate=True,
            args=None
    ):
        if args is not None:
            if args.alg is not None:
                log_dir = os.path.join(log_dir, args.alg)
            if args.map is not None:
                log_dir = os.path.join(log_dir, args.map)
            if args.experiment_name is not None:
                log_name = args.experiment_name + ".log"

        args.run_name = log_name.replace(".log", "")

        if os.path.exists(log_dir) is False:
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, log_name)  # log文件名
        if logger_name is None:
            logger_name = log_path

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = propagate   # smac初始化时会向root添加handler
        self.logger.log_name = log_name
        self.logger.logger_name = logger_name
        formatter = logging.Formatter(fmt, datefmt, style="{")

        if console_log is True:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            sh.setLevel(console_level)
            self.logger.addHandler(sh)

        if file_log is True:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(formatter)
            fh.setLevel(file_level)
            self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def war(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def cri(self, message):
        self.logger.critical(message)

    def starting_log(self, iter_number, args):
        self.logger.info("=" * 60)
        self.logger.info("*" * 60)
        self.logger.info("=" * 60)
        self.logger.info(f"Running {iter_number} run")
        self.logger.info("=" * 60)
        self.logger.info("*" * 60)
        self.logger.info("=" * 60)
        message = ""
        arg_list = list(vars(args).keys())
        arg_list.sort()
        for arg in arg_list:
            message += f"\n{' ' * 4}{arg} = {getattr(args, arg)}"
        self.logger.info("parameters:" + message)
