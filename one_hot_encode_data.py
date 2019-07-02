# TODO: create a dataset defined by the following:
# 1) ID (first column)
# 2) Next, one hot encode the search term
# 3) Finally, one hot encode based on the most popular ingredients

import csv

CATEGORIES = ['beef', 'bread', 'breakfast', 'cheese', 'dessert', 'dimsum', 'fish', 'fruit', 'mushroom', 'noodle', 'pasta', 'pizza', 'pork',
	'potato', 'poultry', 'salad', 'sandwich', 'soup', 'vegetables']

def getNMostCommonIngredients(n):
	most_common_ingredients = []
	with open('counts/part-00000', 'r') as count_file:
		item_count = 0
		while item_count <= n:
			ingredient_count_tuple = eval(count_file.readline())
			ingredient_name = ingredient_count_tuple[0]
			if ingredient_name != ' None' and ingredient_name != 'None':
				most_common_ingredients.append(ingredient_name)
				item_count += 1

	return most_common_ingredients


def writeRows(most_common_ingredients):
	with open('data_one_hot.csv', 'w') as to_file:
		with open('data.csv', 'r') as from_file:
			csv_reader = csv.reader(from_file)
			csv_writer = csv.writer(to_file)

			for row in csv_reader:
				_id = row[0]
				category = row[1]
				one_hot_category = [0 if index != CATEGORIES.index(category) else 1 for index in range(len(CATEGORIES))]
				
				ingredients = eval(row[6])
				one_hot_ingredients = [0 if item not in ingredients else 1 for item in most_common_ingredients]

				write_row = []
				write_row.append(_id)
				write_row.extend(one_hot_category)
				write_row.extend(one_hot_ingredients)

				csv_writer.writerow(write_row)


def main():
	most_common_ingredients = getNMostCommonIngredients(250)
	writeRows(most_common_ingredients)	


if __name__ == "__main__":
	main()