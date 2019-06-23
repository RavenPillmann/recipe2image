#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import tempfile
import json
import re

from . import crf_utils
# import crf_utils

# From https://www.dummies.com/food-drink/recipes/measurement-abbreviations-and-conversions/
ABBRS_TO_UNITS = {
	'c': 'cup',
	'C': 'cup',
	'g': 'gram',
	'kg': 'kilogram',
	'l': 'liter',
	'L': 'liter',
	'lb': 'pound',
	'ml': 'milliliter',
	'mL': 'milliliter',
	'oz': 'ounce',
	'pt': 'pint',
	't': 'teaspoon',
	'tsp': 'teaspoon',
	'T': 'tablespoon',
	'TB': 'tablespoon',
	'Tbl': 'tablespoon',
	'Tbsp': 'tablespoon',

	'cs': 'cups',
	'Cs': 'cups',
	'gs': 'grams',
	'kgs': 'kilograms',
	'ls': 'liters',
	'Ls': 'liters',
	'lbs': 'pounds',
	'mls': 'milliliters',
	'mLs': 'milliliters',
	'ozs': 'ounces',
	'pts': 'pints',
	'ts': 'teaspoons',
	'tsps': 'teaspoons',
	'Ts': 'tablespoons',
	'TBs': 'tablespoons',
	'Tbls': 'tablespoons',
	'Tbsps': 'tablespoons'
}


def removeSpecialCharacters(line):
	# TODO: Remove special characters
	new_line = re.sub('[^a-zA-Z0-9]', ' ', line)
	return new_line


def convertAbbreviationsToUnits(line):
	# TODO: Convert abbreviations to units
	line_split = line.split(" ")
	line_split_abbrs_to_units = [ABBRS_TO_UNITS[word] if (word in ABBRS_TO_UNITS) else word for word in line_split]
	return " ".join(line_split_abbrs_to_units).lower()


def preprocessLines(ingredient_lines):
	for index in range(len(ingredient_lines)):
		new_line = ingredient_lines[index]
		new_line = removeSpecialCharacters(new_line)
		new_line = convertAbbreviationsToUnits(new_line)
		ingredient_lines[index] = new_line

	return ingredient_lines


# TODO: Convert this into code that doesn't require directories or files...
def getParsedIngredients(ingredient_lines):
	ingredient_lines = preprocessLines(ingredient_lines)

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

	print(convertAbbreviationsToUnits('2 lbs rice'))