#!/usr/bin/python


import sys

print "checking for nltk"
try:
    import nltk
except ImportError:
    print "you should install nltk before continuing"
    print "program is exiting..."
    sys.exit(0)

print "checking for numpy"
try:
    import numpy
except ImportError:
    print "you should install numpy before continuing"
    print "program is exiting..."
    sys.exit(0)

print "checking for scipy"
try:
    import scipy
except:
    print "you should install scipy before continuing"
    print "program is exiting..."
    sys.exit(0)

print "checking for sklearn"
try:
    import sklearn
except:
    print "you should install sklearn before continuing"
    print "program is exiting..."
    sys.exit(0)

print "All checks passed"