__author__ = 'Albert Wang'
import logging
from logging.handlers import TimedRotatingFileHandler
import os


class Log:
    def __init__(self, file_path):

        self.logname = os.path.join(file_path, 'log')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        self.formatter = logging.Formatter(
            '[%(asctime)s]  - %(levelname)s: %(message)s')
        tf = TimedRotatingFileHandler(self.logname, "D", 1, 10)
        tf.setFormatter(self.formatter)
        tf.setLevel(logging.DEBUG)

        self.logger.addHandler(tf)

        # 创建一个StreamHandler,用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

    def __out(self, level, message):
        # 创建一个FileHandler，用于写到本地

        if level == 'info':
            self.logger.info(message)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)

        # 关闭打开的文件
        # fh.close()

    def debug(self, message):
        self.__out('debug', message)

    def info(self, message):
        self.__out('info', message)

    def warning(self, message):
        self.__out('warning', message)

    def error(self, message):
        self.__out('error', message)
