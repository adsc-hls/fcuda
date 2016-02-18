#!/usr/bin/python

import apt
import os
import sys

cache=apt.cache.Cache()
cache.update()
cache.open()

# Install all required packages
packages=open('packages.lst','r').read()
for pack in packages.splitlines():
    print "Package {}".format(pack)
    pkg=cache[pack]
    if pkg.is_installed:
        print "{} already installed".format(pack)
    else:
        pkg.mark_install()
        try:
            cache.commit()
        except Exception, arg:
            print >> sys.stderr, "Package installation failed [{err}]".format(err=str(arg))
