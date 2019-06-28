# TODO: 
# Go through each csv
# Give id number for each recipe based on ultimate order
# Store all csv data besides pictures in one csv, id'd by the number
# Download and store all the pictures individually, use the id in their file name so they can be identified.
# Run on google cloud instance, save into bucket

import cv2
import urllib.request
import os
import csv 
import logging
import numpy as np
import sys

logging.basicConfig(filename='formatting_data.log', level=logging.DEBUG)

filenames = os.listdir('data')

index = 0

with open('data.csv', 'w') as df:
	df_writer = csv.writer(df, delimiter=',', quotechar='"')

	for filename in filenames:
		logging.info('Starting %s', filename)

		category = filename[:-4]
		with open('data/'+filename, 'r') as f:
			reader = csv.reader(f, delimiter=",")

			for i, line in enumerate(reader):
				sys.stdout.write("\rIndex %s" % str(i))
				sys.stdout.flush()
				row = []
				row.append(index)
				row.append(category)
				row.append(line[0])
				row.append(line[1])
				row.append(line[2])
				row.append(line[3])

				ingredients = [item.get('name') for item in eval(line[5])]
				row.append(ingredients)

				try:
					resp = urllib.request.urlopen(line[4])
					image = np.asarray(bytearray(resp.read()), dtype="uint8")
					image = cv2.imdecode(image, cv2.IMREAD_COLOR)

					cv2.imwrite('images/' + str(index) + '.jpg', image)
					df_writer.writerow(row)

					index += 1
				
				except urllib.error.URLError as e:
					logging.error('URLError %s', e.code)
					logging.info('Attempted file and index: %s %s', filename, str(i))

				except urllib.error.HTTPError as e:
					logging.error('HTTPError', e.reason)
					logging.info('Attempted file and index: %s %s', filename, str(i))

	logging.info('Successfully got through %s', filename)
