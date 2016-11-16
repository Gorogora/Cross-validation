from CrossValidation import *
from DescensoGradiente import *


def main():
    k = 5
    file_name = "p1_1.csv"
    num_iterations = 100
    alpha = 1.0

    # load and normalize data, and add a column indicating the fold
    data, mu, sigma= load_and_prepare_data(file_name, k)

    # Creamos el fichero para escribir el log
    log = list()
    file = open('./log.txt', 'w')
    inicialize_file(file, file_name, k, num_iterations, alpha)

    # Coger todos los datos. Los que pertenezcan al pliegue que se está examinando se dejan para test
    for fold in range(k):
        fold += 1
        #fold = 1
        training, test = get_training_test_data(data, fold)

        # load  the data
        dataX, dataY, m, num_atrib = load_data(training)

        # inicialize theta
        theta = [0] * num_atrib

        # calculate theta values
        theta = batch_gradient_descent(dataX, dataY, theta, m, num_atrib, alpha, num_iterations)

        # calculate the error with the test dataset
        dataX, dataY, m, num_atrib = load_data(test)
        error = cost_function(dataX,dataY,theta,m)
        log.append(error)


    write_log(log, file)
    file.close()




if __name__ == "__main__":
    main()
