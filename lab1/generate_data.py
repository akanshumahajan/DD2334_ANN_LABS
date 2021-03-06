import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')


def generate_data(N, plot=False, meanA=None, meanB=None, sigmaA=None, sigmaB=None):
    '''
    Generates data of two linearly seperable classes of N samples
    '''

    # Set up gaussian distribution parameters
    if not meanA:
        meanA = [4, 2]
    if not sigmaA:
        covA = np.array([[0.5, 0],
                         [0, 0.5]])
    else:
        covA = np.array([[sigmaA, 0],
                         [0, sigmaA]])
    if not meanB:
        meanB = [-2, 2]
    if not sigmaB:
        covB = np.array([[0.5, 0],
                         [0, 0.5]])
    else:
        covB = np.array([[sigmaB, 0],
                         [0, sigmaB]])

    classA = np.random.multivariate_normal(meanA, covA, N)
    classB = np.random.multivariate_normal(meanB, covB, N)

    classA_extended = np.column_stack([classA, np.ones(N)])
    classB_extended = np.column_stack([classB, -np.ones(N)])
    data = np.row_stack([classA_extended, classB_extended])
    np.random.shuffle(data)
    # data = np.array([[classA, np.zeros(N)],
    #                  [classB, np.ones(N)]])
    # print(data)
    if plot:
        plt.scatter(classA[:, 0], classA[:, 1], label="Class A")
        plt.scatter(classB[:, 0], classB[:, 1], label="Class B")

        plt.plot()
    return data


def generate_data_2(N, plot=False, meanA=None, meanB=None, sigmaA=None, sigmaB=None, special=False):
    '''
    Generates data of two linearly seperable classes of N samples
    '''
    # Set up gaussian distribution parameters
    if not meanA:
        meanA = [4, 2]
    if not sigmaA:
        covA = np.array([[0.5, 0],
                         [0, 0.5]])
    else:
        covA = np.array([[sigmaA, 0],
                         [0, sigmaA]])
    if not meanB:
        meanB = [-2, 2]
    if not sigmaB:
        covB = np.array([[0.5, 0],
                         [0, 0.5]])
    else:
        covB = np.array([[sigmaB, 0],
                         [0, sigmaB]])

    classA = np.random.multivariate_normal(meanA, covA**2, N)
    classB = np.random.multivariate_normal(meanB, covB**2, N)

    if special:
        meanA_alternative = meanA.copy()
        meanA_alternative[0] *= -1
        classA_alternative = np.random.multivariate_normal(
            meanA_alternative, covA**2, N)
        classA[:int(N/2), :] = classA_alternative[:int(N/2), :]

    classA_extended = np.column_stack([classA, np.ones(N)])
    classB_extended = np.column_stack([classB, -np.ones(N)])
    data = np.row_stack([classA_extended, classB_extended])
    np.random.shuffle(data)
    # data = np.array([[classA, np.zeros(N)],
    #                  [classB, np.ones(N)]])
    # print(data)
    if plot:
        plt.scatter(classA[:, 0], classA[:, 1], label="Class A")
        plt.scatter(classB[:, 0], classB[:, 1], label="Class B")
        # plt.show()

    return data
