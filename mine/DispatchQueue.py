import logging
import multiprocessing

def init(queue_len):
    global _dispatch_queue
    _dispatch_queue = multiprocessing.Queue(queue_len)
    _logger = logging.getLogger("minesweaper_log")
    _logger.info("dispatch message queue initialed.")

def put(msg):
    global _dispatch_queue
    _dispatch_queue.put(msg)

def empty():
    global _dispatch_queue
    return _dispatch_queue.empty()

def full():
    global _dispatch_queue
    return _dispatch_queue.full()

def get():
    global _dispatch_queue
    if _dispatch_queue.empty():
        return ()
    return _dispatch_queue.get()

def size():
    global _dispatch_queue
    return _dispatch_queue.qsize()
