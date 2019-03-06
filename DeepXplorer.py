#!/usr/bin/env python

from h5xplorer.h5xplorer import h5xplorer
import menu
import glob
import shutil

app = h5xplorer(menu.context_menu,extended_selection=False)

# clear
# dirs = glob.glob('./_tmp_*')
# for d in dirs:
#     shutil.rmtree(d)