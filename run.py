#!/usr/bin/env python3
"""Entry point:  python run.py"""
from dsbench.config import load_config
from dsbench.runner import run_once


def main():
    run_once(load_config())


if __name__ == "__main__":
    main()