import logging
import os
import sys

# sad hack
for x, arg in enumerate(sys.argv):
    if arg == '-c' and x + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[x + 1]
        break


def fmt_filter(record):
    record.levelname = '[%s]' % record.levelname
    record.funcName = '[%s]' % record.funcName
    record.lineno = '[%s]' % record.lineno
    return True


logging.basicConfig(
    format='%(asctime)s %(levelname)-7s %(name)s %(lineno)-3s: %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().addFilter(fmt_filter)
