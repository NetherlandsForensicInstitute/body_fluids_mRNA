import logging
from os import makedirs
from os.path import join
import git
from datetime import datetime


def create_logdir() -> str:
    """
    creates directory for logging and returns path as string
    :return: string of the logdir
    """
    # get first 8 chars of commit hash of current commit
    repo = git.Repo(search_parent_directories=True)
    repo_hash = repo.head.object.hexsha[:9]
    # get current time
    time_str = '{date:%Y%m%d-%H%M%S}'.format(date=datetime.now())
    # create path for log dir
    log_dir = join('./logs', '_'.join([time_str, repo_hash]))
    # create log dir path
    makedirs(log_dir)

    return log_dir


def create_logging() -> str:
    """
    setup logging and
    :return: string of the logdir
    """
    # Setup base level of logging
    logging.basicConfig(level=logging.INFO)
    # Set name of logger
    logger = logging.getLogger('main')
    # Print to console of
    logger.propagate = False

    # Create and get logdir
    logdir = create_logdir()

    # Write logger to file
    logger.addHandler(logging.FileHandler(join(logdir, 'log.log')))

    # check if repo is clean
    if git.Repo(search_parent_directories=True).is_dirty():
        logger.warning('Not all files were committed before the execution of this script!')

    return logdir
