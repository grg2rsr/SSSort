import sys
import argparse
import subprocess
from pathlib import Path
import platform

if __name__ == '__main__':
    if platform.system() == 'Linux':
        parser = argparse.ArgumentParser(
            description='this is Single Sensillum Sort v.XX authors: Georg Raiser, XX'
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
            '-v', '--verbose', action='store_true', help='print out system call'
        )

        # argument parse
        args = vars(parser.parse_args())
        extra_args = (
            args['extra_args'].split(',') if args['extra_args'] is not None else None
        )

        # paths
        config_path = Path(args['path']).resolve()
        install_dir = Path(sys.argv[0]).parent

        mode = args['mode']

    else:
        mode = sys.argv[1]
        config_path = sys.argv[2]
        install_dir = Path(sys.argv[0]).parent

    if mode == 'convert':
        if extra_args is not None:
            extra_args = ' '.join(extra_args)
            cmd = 'python %s/sssio.py %s %s' % (install_dir, config_path, extra_args)
        else:
            cmd = 'python %s/sssio.py %s' % (install_dir, config_path)

    if mode == 'detect':
        cmd = 'python %s/sssort.py %s %s' % (install_dir, config_path, mode)

    if mode == 'sort':
        cmd = 'python %s/sssort.py %s %s' % (install_dir, config_path, mode)

    if mode == 'inspect':
        cmd = 'python %s/inspect_result.py %s' % (install_dir, config_path)

    if mode == 'merge':
        cmd = 'python %s/manual_merger.py %s' % (install_dir, config_path)

    if mode == 'postprocess':
        cmd = 'python %s/post_processing.py %s' % (install_dir, config_path)

    if args['verbose']:
        print('executing: %s' % cmd)

    subprocess.call(cmd, shell=True)
