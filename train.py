# -*- coding: utf-8 -*-

from modules.pre_process import pre_process
from modules.configuration import TRAIN_FILE


def train():
    pre_process(TRAIN_FILE)


if __name__ == '__main__':
    train()
    