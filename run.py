import sys, os
import argparse
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='this is Single Sensillum Sort v.XX authors: Georg Raiser, XX')
    parser.add_argument("mode", choices=['convert','detect','sort','inspect', 'merge', 'postprocess'], help='determines what to do')
    parser.add_argument("path", help="path to the config file")

    args = vars(parser.parse_args())
    
    config_path = Path(args['path']).resolve()
    mode = args['mode']

    install_dir = Path(sys.argv[0]).parent

    if mode == 'convert':
        subprocess.call("python %s/sssio.py %s " % (install_dir, config_path), shell=True)

    if mode == 'detect':
        subprocess.call("python %s/sssort.py %s %s" % (install_dir, config_path, mode), shell=True)

    if mode == 'sort':
        subprocess.call("python %s/sssort.py %s %s" % (install_dir, config_path, mode), shell=True)

    if mode == 'inspect':
        subprocess.call("python %s/inspect_result.py %s" % (install_dir, config_path), shell=True)

    if mode == 'merge':
        subprocess.call("python %s/manual_merger.py %s" % (install_dir, config_path), shell=True)

    if mode == 'postprocess':
        subprocess.call("python %s/post_processing.py %s" % (install_dir, config_path), shell=True)

