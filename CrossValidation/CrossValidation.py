import random
import csv
from Functions import *

def load_and_prepare_data(file_name, k):
    random.seed(1000)
    # get the path
    path = get_file_path(file_name)
    # open the file
    file = open(path)
    # read the file
    reader = csv.reader(file, delimiter=';')
    # transform data into a list
    x = [list(map(float, line)) for line in reader]
    # numbers of variables
    num_atrib = len(x[0])
    # number of instances of the problem
    m = len(x)
    # shuffle the list
    random.shuffle(x)

    # normalize the data
    temp_x = [elem[:-1] for elem in x]
    temp_x_norm, mu, sigma = normalize(temp_x, num_atrib)
    # add a 1's column
    temp_x = [[1.0] + elem for elem in temp_x_norm]
    # restore the data
    data = [x + [y[-1]] for x,y in zip(temp_x,x)]

    # add a column to the list indicating the fold
    fold = 1
    for i in range(len(x)):
        #rnd = random.randint(1,k)
        data[i].insert(0,fold)
        fold += 1
        if(fold > k):
            fold = 1

    return data, mu, sigma


def get_training_test_data(dataset, fold):
    test = []
    training = []
    for elem in dataset:
        if(elem[0] == fold):
            test.extend([elem[1:]])
        else:
            training.append(elem[1:])

    return training, test
