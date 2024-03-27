import sys, os
import argparse
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='this is Single Sensillum Sort v.XX authors: Georg Raiser, XX')
    parser.add_argument("mode", choices=['convert','detect','sort','inspect'], help='only runs spike detection')
    parser.add_argument("path", help="path to the config file")

    args = vars(parser.parse_args())
    
    path = Path(args['path']).resolve()
    mode = args['mode']

    install_dir = Path(sys.argv[0]).parent

    if mode == 'convert':
        subprocess.call("python %s/sssio.py %s " % (install_dir, path), shell=True)

    if mode == 'detect':
        subprocess.call("python %s/sssort.py %s %s" % (install_dir, path, mode), shell=True)

    if mode == 'sort':
        subprocess.call("python %s/sssort.py %s %s" % (install_dir, path, mode), shell=True)

    if mode == 'inspect':
        subprocess.call("python %s/inspect_result.py %s" % (install_dir, path), shell=True)

    if mode == 'merge':
        # subprocess.call("python %s/inspect_result.py %s" % path, shell=True)
        pass

    if mode == 'postprocess':
        # subprocess.call("python %s/sssort.py %s %s" % (path, mode), shell=True)
        pass

        