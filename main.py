from service.preprocess_service import to_prepare
from utils import logging_utils


def main():
    to_prepare("./dataset/Taipei_QA_new.txt")


if __name__ == '__main__':
    logging_utils.Init_logging()
    main()
