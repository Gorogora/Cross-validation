import os, csv
import numpy as np
import random


def get_file_path(file_name, folder=""):
    if folder == "":
        currentdirpath = os.getcwd()
    else:
        currentdirpath = os.getcwd() + "/" + folder

    file_path = os.path.join(currentdirpath, file_name)

    return file_path


def normalize(dataset, num_atrib):
    dataset_ = np.matrix(dataset)
    mu = np.zeros((1,num_atrib-1))
    sigma = np.zeros((1,num_atrib-1))

    for i in range(num_atrib-1):
        mu[0,i] = np.mean(dataset_[:,i])
        sigma[0,i] = np.std(dataset_[:,i])
        dataset_[:,i] = (dataset_[:,i] - mu[0,i]) / sigma[0,i]

    return dataset_.tolist(), mu, sigma

def inicialize_file(f, file_name, k, num_iterations, alpha):
    f.write('\t*** Log del fichero ' + file_name + ' ***\n')
    f.write('Parámetros empleados:\n')
    f.write('Folds: ' + str(k) + '\n')
    f.write('Número de iteraciones: ' + str(num_iterations) + '\n')
    f.write('Alpha: ' + str(alpha) + '\n\n')
    f.write('\tFOLD\t\tERROR\n')

def write_log(log, f):
    for error, k in zip(log,range(len(log))):
        f.write('\t' + str(k+1) + '\t\t' + str(error) + '\n')

    f.write('\n')
    f.write('Error medio: ' + str(np.mean(log)))

