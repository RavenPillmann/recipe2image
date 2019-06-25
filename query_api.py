# query_api.py
# Raven Pillmann
import urllib.request
import urllib.error
import argparse
import json
import logging
import threading
import csv

import formatters

BASE_URL = 'https://api.edamam.com/search'

logging.basicConfig(filename='query_log.log', level=logging.DEBUG)


def urlBuilder(api_key, api_id, keyword, from_num, to_num):
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
	url += 'from=' + str(from_num) + '&to=' + str(to_num)

	return url


def getData(api_key, api_id, keyword, from_num, to_num):
	"""
	input: 	api_key  - string
			api_id   - string
			keyword  - string
			from_num - string
			to_num	 - string
	return: data - string
	"""
	url = urlBuilder(api_key, api_id, keyword, from_num, to_num)

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

	json_data = json.loads(data)

	for hit in json_data['hits']:
		recipe = hit['recipe']

		ingredient_lines = recipe['ingredientLines']
		parsed_ingredients = eval(formatters.parseIngredient(ingredient_lines))
		recipe['ingredients'] = parsed_ingredients

	return json_data


def storeData(json_data, filename):
	"""
	input: 	json_data - json data from api
			filename  - name of file to save data to
	return: None
	"""
	with open(filename, 'a') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"')

		query_term = json_data.get('q')

		for hit in json_data['hits']:
			recipe = hit['recipe']

			label = recipe['label']
			url = recipe['url']
			source = recipe['source']
			image_url = recipe['image']
			ingredients = recipe['ingredients']

			csv_writer.writerow([query_term, label, url, source, image_url, ingredients])
	

def query(api_key, api_id, keyword, from_num, to_num):
	"""
	input: 	api_key - string
			api_id  - string
			keyword - string
	return: None
	"""
	res = getData(api_key, api_id, keyword, from_num, to_num)

	recipes_with_individual_ingredients = res

	if len(res) > 0:
		recipes_with_individual_ingredients = postprocessData(res)

	return recipes_with_individual_ingredients


def getAndStoreData(api_key, api_id, keyword, from_num, to_num, output_file):
	res = query(api_key, api_id, keyword, from_num, to_num)
	
	if len(res['hits']):
		storeData(res, output_file)
		return 1
	else:
		return 0


def main():
	parser = argparse.ArgumentParser(description="Query the EDAMAM recipe api.")

	parser.add_argument('--apikey', '-a', dest="APIKey")
	parser.add_argument('--apiid', '-ai', dest="APIId")
	parser.add_argument('--keyword', '-k' , dest="Keyword")
	parser.add_argument('--from', '-f', dest="FromNum", default=0)
	parser.add_argument('--to', '-t', dest="ToNum", default=10)
	parser.add_argument('--output_file', '-o', dest="OutputFile", default='test.csv')

	args = parser.parse_args()
	args = vars(args)

	res = getAndStoreData(args['APIKey'], args['APIId'], args['Keyword'], args['FromNum'], args['ToNum'], args['OutputFile'])


if __name__ == "__main__":
	main()
