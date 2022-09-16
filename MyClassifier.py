from sys import argv
from math import sqrt
from math import exp
from math import pi
"""
Function to open training data file.

Params: 1) @name: file containing training data
Returns:1) data in matrix form
	   all attribrute values are converted to float
	   yes: 1.0, no: 0.0
"""
def openTrainingFile(name):

	data_matrix = []
	testFile = open(name, "r")
	lines = testFile.readlines()

	"Split lines and remove \n"
	for i in range(0, len(lines)):
		lines[i] = (lines[i].rstrip("\n")).split(",")
	

	for line in lines:
		
		if len(line) == 1:
			break

		for i in range(0, len(line)):

			if line[i] == "yes":
				line[i] = 1.0
			elif line[i] == "no":
				line[i] = 0.0
			else:	
				line[i] = float(line[i])
		
		data_matrix.append(line)
	
	return data_matrix


"""
Function to open test data file.

Params: 1) @name: file containing test data

Returns:1) data in matrix form 
	   all attribrute values are converted to float
	   
"""
def openTestFile(name):
	
	data_matrix = []
	testFile = open(name, "r")
	lines = testFile.readlines()

	"Split lines and remove \n"
	for i in range(0, len(lines)):
		lines[i] = (lines[i].rstrip("\n")).split(",")
	

	for line in lines:
		
		if len(line) == 1:
			break

		for i in range(0, len(line)):

			line[i] = float(line[i])
		
		data_matrix.append(line)
	
	return data_matrix


"""
Calculates Euclidian Distance

@Params: 1) point1: array of floats
	 2) point2: array of floats

@Return: Euclidian distance
"""
def EuclidianDistance(point1, point2):
	d = 0.0
	for i in range(0, len(point2)):
		d += (point1[i] - point2[i])**2

	return sqrt(d)

	
"""
Runs KNN algorithm for test data

@Params: 1) data : training data (in matrix format)
	 2) inp: test data input (array of floats)
	 3) k: Number of closest neighbours (given by user)
@Result: Prints yes or no to stdout 
"""
def KNN(data, inp, k): 
	
	distance_list = list() #Stores all euclidian distance with point

	for row in data:
		distance_list.append((row ,EuclidianDistance(row, inp))) #stores (test_row, dist)

	distance_list.sort(key=lambda entry: entry[1])	#Sort by distance
	
	neighbours = list()
	i = 0
	while i != k:
		neighbours.append(distance_list[i][0])	#Append first k neighbours after sort
		i += 1
	

	# Calculating class count for k nearest neighbours

	classifications = [data[-1] for data in neighbours]
	count_yes = 0
	count_no = 0
	for classification in classifications:
		if classification == 0.0:
			count_no += 1
		elif classification == 1.0:
			count_yes += 1
	
	if count_no > count_yes:
		result = 0.0

	else:
		result = 1.0

        # Printing output
	
	if result == 1.0:
		print("yes")

	elif result == 0.0:
		print("no")



"""Returns dictonary of fors {class_value:frequence, ...}"""
def create_class_dict(data):

	class_dict = {}

	for i in range(len(data)):
		class_of_item = data[i][-1]
		if  (class_of_item not in class_dict):
			class_dict[class_of_item] = list()
		class_dict[class_of_item].append(data[i])

	return class_dict


def mean(list_of_numbers):
	return float(sum(list_of_numbers) / (len(list_of_numbers)))

def stdev(list_of_numbers):
	avg = mean(list_of_numbers)
	variance = sum([(x-avg)**2 for x in list_of_numbers]) / float(len(list_of_numbers) - 1)
	return sqrt(variance)


"""Returns a list of tuples of form [(mean, stdev, len(column) ), ...] for all columns in the dataset"""
def get_dataset_summary(data):
	out = [(mean(column), stdev(column), len(column)) for column in zip(*data)]
	del(out[-1])
	return out


def summary_by_class(data):
	class_separated_data = create_class_dict(data)
	summary = dict()
	for outcome_class, corresponding_rows in class_separated_data.items():
		summary[outcome_class] = get_dataset_summary(corresponding_rows)
	return summary

def calculate_gaussian_probability(instance, mean, stdev):
	e = exp(-((instance-mean)**2 / (2 * stdev**2)))
	return (1 / (sqrt(2 *pi) * stdev)) * e



def get_class_probabilities(summaries, instance):
	total_rows = sum([len(x) for x in summaries.values()])
	probability_dict = dict()
	for outcome_class, summary in summaries.items():
		probability_dict[outcome_class] = summaries[outcome_class][0][2]/float(total_rows)
		for i in range(len(summary)):
			mean, stdev, length = summary[i]
			probability_dict[outcome_class] *= calculate_gaussian_probability(instance[i], mean, stdev)
	return probability_dict


def predictNB(summaries, instance):
	probabilities = get_class_probabilities(summaries, instance)
	outcome_class = None
	best_prob = -1
	for class_value, probability in probabilities.items():
		if outcome_class is None or probability > best_prob:
			best_prob = probability
			outcome_class = class_value
	return outcome_class

def NB(training, test):
	summaries_by_class = summary_by_class(training)
	outcomes = []
	for instance in test:
		out = predictNB(summaries_by_class, instance)
		if out == 1.0:
			print("yes")

		elif out == 0.0:
			print("no")
		outcomes.append(out)
	return outcomes
	
"""
main
"""

training_name = argv[1]
test_name = argv[2]
algorithm = argv[3]

training_data = openTrainingFile(training_name)
test_data = openTestFile(test_name)



if algorithm == "NB":
     #NB ALGO
     NB(training_data, test_data)
     

else:
     #KNN ALGO
     for test_subject in test_data:
     	KNN(training_data, test_subject, int(algorithm[0]))	






