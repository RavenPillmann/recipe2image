# query_api.py
# Raven Pillmann
import urllib.request
import urllib.error
import argparse
import json
import logging
import threading

import formatters

BASE_URL = 'https://api.edamam.com/search'

logging.basicConfig(filename='query_log.log', level=logging.DEBUG)


def urlBuilder(api_key, api_id, keyword):
	"""
	input: 	api_key - string
			api_id  - string
			keyword - string
	return: built/formatted endpoint url
	"""
	url = BASE_URL + '?'

	url += 'app_id=' + str(api_id) + '&'
	url += 'app_key=' + str(api_key) + '&'
	url += 'q=' + str(keyword) + '&'
	url += 'from=0&to=10'

	return url


def query(api_key, api_id, keyword):
	"""
	input: 	api_key - string
			api_id  - string
			keyword - string
	return: data - string
	"""
	url = urlBuilder(api_key, api_id, keyword)

	req = urllib.request.Request(url)
	
	data = ""

	try:
		with urllib.request.urlopen(req) as response:
			data = response.read()

	except urllib.error.URLError as e:
		logging.error('URLError %s', e.code)
		logging.info('Attempted Keyword: %s', keyword)
	except urllib.error.HTTPError as e:
		logging.error('HTTPError', e.reason)
		logging.info('Attempted Keyword: %s', keyword)

	return data


def postprocessData(data):
	"""
	input: 	data - string
	return: formatted json data
	"""

	# TODO: parse the ingredients, put parsed information into ingredients
	json_data = json.loads(data)

	# recipes_with_individual_ingredients = list(filter(lambda x: 'food' in x['recipe']['ingredients'][0], json_data['hits']))
	for hit in json_data['hits']:
		recipe = hit['recipe']

		for ingredient in recipe['ingredients']:
			if 'food' not in ingredient:
				text = ingredient['text']

				parsed_text = eval(formatters.parseIngredient([text]))[0]
				# print("parsed_text", parsed_text)
				ingredient['food'] = parsed_text.get('name')
				ingredient['unit'] = parsed_text.get('unit')
				ingredient['qty'] = parsed_text.get('qty')


	return json_data


def queryAndStore(api_key, api_id, keyword):
	"""
	input: 	api_key - string
			api_id  - string
			keyword - string
	return: None
	"""
	res = query(api_key, api_id, keyword)

	recipes_with_individual_ingredients = res

	if len(res) > 0:
		recipes_with_individual_ingredients = postprocessData(res)

	return recipes_with_individual_ingredients


def main():
	parser = argparse.ArgumentParser(description="Query the EDAMAM recipe api.")

	parser.add_argument('--apikey', '-a', dest="APIKey")
	parser.add_argument('--apiid', '-ai', dest="APIId")
	parser.add_argument('--keyword', '-k' , dest="Keyword")

	args = parser.parse_args()
	args = vars(args)

	# print(formatters.parseIngredient(['2 ounces dry roasted cherries']))
	res = queryAndStore(args['APIKey'], args['APIId'], args['Keyword'])
	# print("postprocessed results", res)


if __name__ == "__main__":
	main()
