import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler

# 日志级别
CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(CURRENT_PATH, os.pardir)
LOG_PATH = os.path.join(ROOT_PATH, 'logs')



# def Singleton(cls):
#     instance = {}
#     def wapper():
#         if cls not in instance:
#             instance[cls] = cls(*args, **kwargs)
#         return instance[cls]
#     return wapper


class LogHandler(logging.Logger):
    """
    LogHandler
    """
    
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            # 一开始居然用了 cls()来实例化 导致无限次调用
            # cls._instance = cls(*args, **kwargs)
            # cls._instance = object.__new__(cls, *args, **kwargs)
            cls._instance = object.__new__(cls)
        return cls._instance


    def __init__(self, name='log', level=DEBUG, stream=True, file=True):
        # super(LogHandler, self).__init__(name=name, level=level)
        self.name = name
        self.level = level
        if len(os.listdir(LOG_PATH)) > 0:
            os.system('rm {}/*.log'.format(LOG_PATH))
        logging.Logger.__init__(self, self.name, level=level)
        if stream:
            self.__setStreamHandler__()
        if file:
            self.__setFileHandler__()

    def __setFileHandler__(self, level=None):
        """
        set file handler
        :param level:
        :return:
        """
        file_name = os.path.join(LOG_PATH, '{name}.log'.format(name=self.name))
            
        # 设置日志回滚, 保存在log目录, 一天保存一个文件, 保留15天
        file_handler = TimedRotatingFileHandler(filename=file_name, when='D', interval=1, backupCount=15)
        file_handler.suffix = '%Y%m%d.log'
        if not level:
            file_handler.setLevel(self.level)
        else:
            file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

        file_handler.setFormatter(formatter)
        self.file_handler = file_handler
        self.addHandler(file_handler)

    def __setStreamHandler__(self, level=None):
        """
        set stream handler
        :param level:
        :return:
        """
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        stream_handler.setFormatter(formatter)
        if not level:
            stream_handler.setLevel(self.level)
        else:
            stream_handler.setLevel(level)
        self.addHandler(stream_handler)

    def resetName(self, name):
        """
        reset name
        :param name:
        :return:
        """
        self.name = name
        self.removeHandler(self.file_handler)
        self.__setFileHandler__()


if __name__ == '__main__':
    # log = LogHandler('test')
    # log.info('this is a test msg')


    logger1 = LogHandler('logger')
    logger2 = LogHandler('logger')
    print(logger1, id(logger1), logger2, id(logger2))
    logger1.info('haha')
    logger2.info('hehe')