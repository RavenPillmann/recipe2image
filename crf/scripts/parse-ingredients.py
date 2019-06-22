#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import tempfile

import crf_utils

if len(sys.argv) < 2:
    sys.stderr.write('Usage: parse-ingredients.py FILENAME')
    sys.exit(1)

FILENAME = str(sys.argv[1])

# TODO: Convert this into code that doesn't require directories or files...

_, tmpConversionFile = tempfile.mkstemp()

with open(FILENAME) as infile, open(tmpConversionFile, 'w') as outfile:
    outfile.write(crf_utils.export_data(infile.readlines()))

tmpFilePath = "../tmp/model_file"
modelFilename = os.path.join(os.path.dirname(__file__), tmpFilePath)
# print(modelFilename, tmpFile)
# print("crf_test -v 1 -m %s %s" % (modelFilename, tmpFile))
output = os.popen("crf_test -v 1 -m %s %s" % (modelFilename, tmpConversionFile)).read()
os.system("rm %s" % tmpConversionFile)
print("out", output)



# os.system("python ./convert-to-json.py %s" % (tmpOutputFile))
# os.system("rm %s" % tmpOutputFile)
