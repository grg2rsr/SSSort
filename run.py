import sys
import argparse
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='this is Single Sensillum Sort v.XX authors: Georg Raiser')
    parser.add_argument("mode", choices=['convert','detect','sort','inspect'], help='only runs spike detection')
    parser.add_argument("path", help="path to the config file")

    args = vars(parser.parse_args())
    
    path = Path(args['path']).resolve()
    mode = args['mode']

    if mode == 'convert':
        subprocess.call("python /home/georg/code/SSSort/sssio.py %s " % path, shell=True)

    if mode == 'detect':
        subprocess.call("python /home/georg/code/SSSort/sssort.py %s %s" % (path, mode), shell=True)

    if mode == 'sort':
        subprocess.call("python /home/georg/code/SSSort/sssort.py %s %s" % (path, mode), shell=True)

    if mode == 'inspect':
        subprocess.call("python /home/georg/code/SSSort/inspect_result.py %s" % path, shell=True)
        pass

    if mode == 'postprocess':
        # subprocess.call("python /home/georg/code/SSSort/sssort.py %s %s" % (path, mode), shell=True)
        pass

        