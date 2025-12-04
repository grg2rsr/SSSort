import argparse
from pathlib import Path
import os

def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='this is Single Sensillum Sort v.XX authors: Georg Raiser, XX',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )
    parser.add_argument(
        'mode',
        choices=['convert', 'detect', 'sort', 'inspect', 'merge', 'postprocess'],
        help='determines what to do',
    )
    parser.add_argument('path', help='path to the config file')

    parser.add_argument(
        '-e',
        '--extra-args',
        action='store',
        dest='extra_args',
        help='optional extra arguments, comma separated',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='print verbose output' # this could be influencing the logger
    )

    return parser

def run(**kwargs):
    mode = kwargs['mode']
    src_folder = Path(__file__).parent.resolve()
    config_path = Path(kwargs['path']).resolve()

    if mode == 'convert':
        if 'extra_args' in kwargs:
            extra_args = ' '.join(kwargs['extra_args'])
            cmd = f"python {src_folder / 'sssio.py'} {config_path} {extra_args}"
        else:
            cmd = f"python {src_folder / 'sssio.py'} {config_path}"

    if mode == 'detect':
        cmd = f"python {src_folder /'sssort.py'} {config_path} {mode}"

    if mode == 'sort':
        cmd = f"python {src_folder / 'sssort.py'} {config_path} {mode}"

    if mode == 'inspect':
        cmd = f"python {src_folder / 'inspect_result.py'} {config_path}"

    if mode == 'merge':
        cmd = f"python {src_folder / 'manual_merger.py'} {config_path}"

    if mode == 'postprocess':
        cmd = f"python {src_folder / 'post_processing.py'} {config_path}"

    if 'verbose' in kwargs:
        print(f'executing: {cmd}')

    os.system(cmd)

def main_cli():
    args = _arg_parser().parse_args()
    run(**vars(args), interactive=True)