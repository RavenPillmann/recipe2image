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
	url += 'q=' + str(keyword)
	url += 'from=0&to=100'

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

	recipes_with_individual_ingredients = list(filter(lambda x: 'food' in x['recipe']['ingredients'][0], json_data['hits']))
	for hit in json_data['hits']:
		recipe = hit['recipe']

		for ingredient in recipe['ingredients']:
			if food not in ingredient:
				text = ingredient['text']

				parsed_text = formatters.parseIngredient(text)
				ingredient['food'] = parsed_text['']


	return recipes_with_individual_ingredients


def queryAndStore(api_key, api_id, keyword):
	"""
	input: 	api_key - string
			api_id  - string
			keyword - string
	return: None
	"""
	res = query(api_key, api_id, keyword)

	if len(res) > 0:
		recipes_with_individual_ingredients = postprocessData(res)
		# print(len(recipes_with_individual_ingredients))


# def printAndDelete(arr):
# 	while len(arr) > 0:
# 		print(arr.pop(0))


# TODO: multithread
# Keep queue of things to query by appearances so far
# Keep a dictionary of term: isQueried to ensure I don't get the same stuff over and over
# Use the url to recipe to not save the same thing twice
def main():
	parser = argparse.ArgumentParser(description="Query the EDAMAM recipe api.")

	parser.add_argument('--apikey', '-a', dest="APIKey")
	parser.add_argument('--apiid', '-ai', dest="APIId")
	parser.add_argument('--keyword', '-k' , dest="Keyword")

	args = parser.parse_args()
	args = vars(args)

	# queryAndStore(args['APIKey'], args['APIId'], args['Keyword'])
	print(formatters.parseIngredient(['2 ounces dry roasted cherries']))

	# arr = ['hello', 'I', 'am', 'a', 'man']
	# thread_1 = threading.Thread(target=printAndDelete, args=(arr,))
	# thread_2 = threading.Thread(target=printAndDelete, args=(arr,))

	# thread_1.start()
	# thread_2.start()

	# thread_1.join()
	# thread_2.join()

	# print("done")


if __name__ == "__main__":
	main()
