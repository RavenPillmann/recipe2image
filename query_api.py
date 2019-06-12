# query_api.py
# Raven Pillmann
import urllib.request
import urllib.error
import argparse
import json
import logging

BASE_URL = 'https://api.edamam.com/search'

logging.basicConfig(filename='query_log.log', level=logging.DEBUG)

def urlBuilder(api_key, api_id, keyword):
	url = BASE_URL + '?'

	url += 'app_id=' + str(api_id) + '&'
	url += 'app_key=' + str(api_key) + '&'
	url += 'q=' + str(keyword)
	# url += 'from=0&to=10' 					# Make 0 to 100?

	return url


def query(api_key, api_id, keyword):
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
	json_data = json.loads(data)
	# print(json_data['hits'][0])

	recipes_with_individual_ingredients = list(filter(lambda x: 'food' in x['recipe']['ingredients'][0], json_data['hits']))

	return recipes_with_individual_ingredients


def main():
	parser = argparse.ArgumentParser(description="Query the EDAMAM recipe api.")

	parser.add_argument('--apikey', '-a', dest="APIKey")
	parser.add_argument('--apiid', '-ai', dest="APIId")
	parser.add_argument('--keyword', '-k' , dest="Keyword")

	args = parser.parse_args()
	args = vars(args)
	res = query(args['APIKey'], args['APIId'], args['Keyword'])

	if len(res) > 0:
		recipes_with_individual_ingredients = postprocessData(res)
		# print(len(recipes_with_individual_ingredients))


if __name__ == "__main__":
	main()