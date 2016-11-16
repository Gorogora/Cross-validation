# Hypothesis function
def h(x, theta):
    return sum([theta_i * x_i for theta_i, x_i in zip(theta, x)])

# Cost function
def cost_function(x, y, theta, m):
    return (0.5 / m) * sum([(h(x_i, theta) - y_i) ** 2 for x_i, y_i in zip(x, y)])

def batch_gradient_descent(dataX, dataY, theta, m, num_atrib, alpha, num_iterations):
    for it in range(num_iterations):
        # update theta
        temp_theta = [0] * num_atrib
        for j in range(num_atrib):
            grad = (1 / m) * sum([(h(x_i, theta) - y_i) * x_i[j] for x_i, y_i in zip(dataX, dataY)])
            temp_theta[j] = theta[j] - alpha * grad

        theta = temp_theta

    return theta

def load_data(dataset):
    # load inputs
    temp_datax = [elem[:-1] for elem in dataset]  # return [[2100.0, 3.0], [1600.0, 3.0], [..., ...]]
    # load outputs
    temp_datay = [elem[-1] for elem in dataset]  # return [400000.0, 330000.0, 369000.0, 232000.0, ... ,]
    # number of instances of the problem
    m = len(temp_datay)
    # numbers of variables + 1
    num_atribu = len(temp_datax[0])

    return temp_datax, temp_datay, m, num_atribu  # return a tuple, we could also write (tempDataX, tempDataY, ...)

