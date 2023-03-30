import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S')

def log(message):
    logging.info(message)
    return 

if __name__ == '__main__':
    logging.info('?')