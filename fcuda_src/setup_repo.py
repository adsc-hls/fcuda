#!/usr/bin/python

import subprocess
import os
import re
import sys
import getopt
import shutil

def checkout_or_update(remote, dir) :
  if not os.path.exists(dir):
    subprocess.call(["git", "clone", "--progress", remote, dir])
  else :
    os.chdir(dir)
    subprocess.call(["git", "remote", "update"])

def main(argv):     
    # Please source the script: env_vars.sh at the top-level directory
    fcuda_dir = os.environ['FCUDA_DIR'] 
    print fcuda_dir
    fcuda_benchmarks = os.path.join(fcuda_dir, "fcuda-benchmarks")
    fcuda_backend = os.path.join(fcuda_dir, "fcuda-backend")
    # Get FCUDA benchmarks repository
    checkout_or_update("https://github.com/adsc-hls/fcuda-benchmarks.git", fcuda_benchmarks)
    # Get FCUDA backend repository
    checkout_or_update("https://github.com/adsc-hls/fcuda-backend.git", fcuda_backend)
    
if __name__ == "__main__":
   main(sys.argv[1:])
