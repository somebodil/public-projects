import logging

logger = logging.getLogger(__name__)


def log_init(log_level=logging.INFO):
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        level=log_level,
    )


def log_args(used_args, unused_args):
    if used_args:
        logger.info('[List of used arguments]')
        used_args_sorted_key = sorted(used_args.__dict__)
        for key in used_args_sorted_key:
            logger.info(f'{key}: {used_args.__dict__[key]}')

    if unused_args:
        logger.info(f'[List of unused arguments]: {unused_args}')
