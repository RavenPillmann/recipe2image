#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import tempfile
import json

from . import crf_utils

# TODO: Convert this into code that doesn't require directories or files...
def getParsedIngredients(ingredient_lines):
	_, tmpConversionFile = tempfile.mkstemp()

	with open(tmpConversionFile, 'w') as outfile:
		outfile.write(crf_utils.export_data(ingredient_lines))

	tmpFilePath = "../tmp/model_file"
	modelFilename = os.path.join(os.path.dirname(__file__), tmpFilePath)
	# print(modelFilename, tmpFile)
	# print("crf_test -v 1 -m %s %s" % (modelFilename, tmpFile))
	output = os.popen("crf_test -v 1 -m %s %s" % (modelFilename, tmpConversionFile)).read()
	os.system("rm %s" % tmpConversionFile)

	output = output.split("\n")
	output_json = json.dumps(crf_utils.import_data(output), indent=4)

	return output_json


if __name__ == '__main__':
	if len(sys.argv) < 2:
	    sys.stderr.write('Usage: parse-ingredients.py FILENAME')
	    sys.exit(1)

	FILENAME = str(sys.argv[1])
	ingredient_lines = open(FILENAME).readlines()

	print(getParsedIngredients(ingredient_lines))