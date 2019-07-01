# TODO: create a dataset defined by the following:
# 1) ID (first column)
# 2) Next, one hot encode the search term
# 3) Finally, one hot encode based on the most popular ingredients

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


def writeRows():
	# TODO: For each row, place ID, then place one out of however many query terms, finally one-hot encode ingredients
	pass


def main():
	most_common_ingredients = getNMostCommonIngredients(250)	


if __name__ == "__main__":
	main()