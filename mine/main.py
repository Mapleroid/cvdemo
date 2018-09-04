import logging
import multiprocessing

def logInit():
    logger = logging.getLogger("minesweaper_log")
    logger.setLevel(level = logging.DEBUG)

    # set file log level to info
    file_handler = logging.handlers.RotatingFileHandler('mine.log', maxBytes=10*1024*1024,backupCount=3)
    file_handler.setLevel(logging.INFO)
    file_handler_formatter = logging.Formatter('%(asctime)s - %(filename)s, %(lineno)d - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_handler_formatter)
    logger.addHandler(file_handler)

    # set console log level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler_formatter = logging.Formatter('%(asctime)s - %(filename)s, %(lineno)d - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_handler_formatter)
    logger.addHandler(console_handler)

def start():
    # init logging
    logInit()
    _logger = logging.getLogger("minesweaper_log")
    _logger.info("program starting...")

    # init queue for multiprocessing
    DispatchQueue.init(6)

    pool = multiprocessing.Pool(processes= 2 + 6)

    # start http server

    pool.apply_async(Worker.start_work)
    pool.apply_async(Worker.start_work)
    pool.apply_async(Worker.start_work)
    pool.apply_async(Worker.start_work)

    pool.close()
    pool.join()


if __name__ == "__main__":
    start()