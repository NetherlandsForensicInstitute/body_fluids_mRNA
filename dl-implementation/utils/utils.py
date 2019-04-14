import logging
import time
from os import makedirs
from os.path import join
import git


def create_logdir() -> str:
    """
    creates directory for logging and returns path as string
    :return: string of the logdir
    """
    repo = git.Repo(search_parent_directories=True)
    # create path for log dir
    log_dir = join('./logs', '_'.join([str(time.time()).replace('.', ''), repo.head.object.hexsha]))
    # create log dir path
    makedirs(log_dir)

    return log_dir


def create_logging() -> str:
    """
    setup logging and
    :return: string of the logdir
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main')
    logger.propagate = False

    logdir = create_logdir()

    logger.addHandler(logging.FileHandler(join(logdir, 'log.log')))

    if git.Repo(search_parent_directories=True).is_dirty():
        logger.warning('Not all files were committed before the execution of this script!')

    return logdir
