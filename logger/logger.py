import os
import logging

class Logger:
    def __init__(self, log_dir, log_file='training.log'):
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, log_file)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                             handlers=[
                                 logging.FileHandler(log_file_path),
                                 logging.StreamHandler()
                                 ]
                            )
        self.logger = logging.getLogger()

    def log(self, message):
        self.logger.info(message)