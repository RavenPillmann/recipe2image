# formatter utilities
# Raven Pillmann
"""
This file is intended to hold formatting utilities, particularly for processing data
"""

from ingredient_parser.en import parse

def parseIngredient(ingredient_string):
	return parse(ingredient_string)
