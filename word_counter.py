import argparse
import os
from pyspark import SparkContext, SparkConf, SQLContext


def countWords(sc, input_file):
	os.system("rm -rf counts") # Remove previous counts to avoid error

	csv_file = sc.textFile(input_file)

	counts = csv_file.map(lambda line: line.split("[")[1]) \
			.map(lambda line: line.split("]")[0]) \
			.flatMap(lambda line: line.split(",")) \
			.map(lambda word: word.strip("\"")) \
			.map(lambda word: word.strip()) \
			.map(lambda word: word.strip("'")) \
			.filter(lambda word: word != ' None' and word != 'None') \
			.map(lambda word: (word, 1)) \
			.reduceByKey(lambda word_1, word_2: word_1 + word_2) \
			.sortBy(lambda word_tuple: word_tuple[1], ascending=False)

	counts.saveAsTextFile("counts")


def main():
	parser = argparse.ArgumentParser(description="Count words")

	parser.add_argument("--input_file", "-i", dest="input_file", help="Input file (data.csv)")
	parser.add_argument("--mode", "-m", dest="mode", help="The spark mode", default="local")

	args = parser.parse_args()
	args = vars(args)

	conf = SparkConf().setAppName("word_counter").setMaster(args['mode'])
	sc = SparkContext(conf=conf)

	countWords(sc, args['input_file'])


if __name__ == "__main__":
	main()