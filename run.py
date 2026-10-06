#!/usr/bin/env python3
"""Entry point:  python run.py [options]     (python run.py -h lists every option)"""
from dsbench.config import parse_args
from dsbench.runner import compare, run_once


def main():
    a = parse_args()
    compare(a) if a.compare else run_once(a)


if __name__ == "__main__":
    main()
