# import sys
# import os
# from pathlib import Path
# import shutil

# install_dir = Path(sys.argv[0]).resolve().parent
# user = os.getlogin()

# out_lines = []
# with open('sssort.sh', 'r') as fH:
#     for line in fH.readlines():
#         if line.startswith('INSTALL_DIR'):
#             out_lines.append('INSTALL_DIR=%s\n' % install_dir)
#         else:
#             out_lines.append(line)

# bin_dir = Path('/home/%s/.local/bin' % user)
# os.makedirs(bin_dir, exist_ok=True)

# with open('/home/%s/.bashrc' % user, 'r') as fH:
#     bashrc = fH.readlines()

# if 'export PATH=$PATH:/home/%s/.local/bin\n' % user not in bashrc:
#     bashrc.append('# added by SSSort installer #\n')
#     bashrc.append('export PATH=$PATH:/home/%s/.local/bin\n' % user)

#     # make a backup of .bashrc
#     shutil.copy2('/home/%s/.bashrc' % user, '/home/%s/.bashrc.bak' % user)

#     with open('/home/%s/.bashrc' % user, 'w') as fH:
#         fH.writelines(bashrc)


# with open(bin_dir / 'sssort', 'w') as fH:
#     fH.writelines(out_lines)

# os.system('chmod u+x %s' % (bin_dir / 'sssort'))
# # os.system('source /home/%s/.bashrc' % user)
