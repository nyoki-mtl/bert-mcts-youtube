from argparse import ArgumentParser
from pathlib import Path

from cshogi import cli


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('engine_path')
    return parser.parse_args()


def main(args):
    benchmark = str(Path('./work_dirs/LesserkaiSrc/Lesserkai/Lesserkai').resolve())
    engine = str((Path(args.engine_path)).resolve())

    print('benchmark: ', benchmark)
    print('engine: ', engine)

    cli.main(benchmark, engine)


if __name__ == '__main__':
    args = parse_args()
    main(args)
