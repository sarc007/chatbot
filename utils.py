from treelib import Node, Tree
from itertools import permutations


def get_combinations(sample_keywords:list):
	'''
		Reveals all combinations of keywords.
	'''
	
	combinations =[]
	for n in range(len(sample_keywords)+1, 0,-1):
		# if n==1:
		# 	list_keywords = [x for x in permutations(sample_keywords, n)]
		# 	combinations.append(list_keywords)	
		# else:
		if n==1 :
			list_keywords = [combinations.append(x[0]) for x in permutations(sample_keywords, n)]
			
		else:
			list_keywords = [ list(x) for x in permutations(sample_keywords, n)]
			for keyword in list_keywords:
				combine_string = ''
				for word in keyword:
					if combine_string == '':
						combine_string = word
					else:
						combine_string += ' '+word
				combinations.append(combine_string)
				
		
			# combinations.append(list_keywords)	
		# print(list_keywords)
		# combinations.append(list_keywords)	
	
	return combinations
	
# combinations =[]
sample_keywords =['get', 'library', 'services']
# sample_factorial = factorial(len(sample_keywords))
# print(sample_factorial)
# print(combinations)
# print(set(permutations(sample_keywords,n)))
print(len(get_combinations(sample_keywords)))
print(get_combinations(sample_keywords))
# for combination in combinations:
