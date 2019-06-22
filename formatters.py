# formatter utilities
# Raven Pillmann
"""
This file is intended to hold formatting utilities, particularly for processing data
"""

from ingredient_parser.en import parse
from crf.scripts import parse_ingredients

def parseIngredient(ingredient_string):
	# return parse(ingredient_string)
	return parse_ingredients.getParsedIngredients(ingredient_string)


if __name__ == "__main__":
	sample_input = ['1 pound carrots']
	print("sample input", sample_input)
	print(parseIngredient(sample_input))