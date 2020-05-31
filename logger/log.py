import os
import logging

import config


# TAKEN FROM: https://docs.python.org/3/howto/logging-cookbook.html


def set_logger(log_path):
    dirs, file = os.path.split(log_path)
    os.makedirs(dirs, exist_ok=True)

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)-12s %(levelname)-8s %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        filename=log_path,
                        filemode='a')
    config.log = logging.getLogger()
    if config.args.verbose:
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        config.log.addHandler(console)
