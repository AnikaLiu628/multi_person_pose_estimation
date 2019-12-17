import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def main():
    model_name = 'MPPE_MOBILENET_V1_0.5_360_640_v2'
    path_root = '../logs/{}.log'.format(model_name)

    training_data = []
    validation_data = []
    evaluation_data = []
    pretrained_model_steps = 13109116
    line_count = 0
    with open(path_root, 'r') as f:
        for line in f:
            if 'Evaluation Loss' in line:
                line_count += 1
                row = line.split(',')
                acc = 0
                for i, col in enumerate(row):
                    if i == 3:
                        col = col.split('(')[0]
                    if 'Loss' in col:
                        loss = float(col.split('=')[-1])
                    # elif 'Error' in col:
                    #     err = float(col.split('=')[-1])
                    elif 'Steps' in col:
                        steps = int(col.split('=')[-1])
                    elif 'Rate' in col:
                        lr = float(col.split('=')[-1])
                    elif 'Batch Size' in col:
                        bs = int(col.split('=')[-1])
                if bs != 256:
                    evaluation_data.append([steps, loss, lr])
                    line_count = 0
            elif 'Training Loss' in line:
                row = line.split(',')
                for i, col in enumerate(row):
                    if i == 2:
                        col = col.split('(')[0]
                    if 'Loss' in col:
                        loss = float(col.split('=')[-1])
                    # elif 'Error' in col:
                    #     err = float(col.split('=')[-1])
                    # elif 'Intensities' in col:
                    #     pass
                    elif 'Steps' in col:
                        steps = int(col.split('=')[-1])
                    elif 'Rate' in col:
                        lr = float(col.split('=')[-1])
                training_data.append([steps, loss, lr])

    training_data = np.array(training_data)
    validation_data = np.array(validation_data)
    evaluation_data = np.array(evaluation_data)
    print('Training Data:', training_data.shape)
    print('Evaluation Data:', evaluation_data.shape)

    #f = interp1d(training_data[:, 0], training_data[:, 1], kind='cubic')
    #xnew = np.linspace(1, 19000, num=20, endpoint=True)

    P = plt.figure(1)

    p1 = plt.subplot(211)
    #plt.plot(training_data[:, 0], training_data[:, 1], 'o')
    #plt.plot(xnew, f(xnew), '-')
    plt.plot(training_data[100:, 0], training_data[100:, 1], 'b')
    plt.plot(evaluation_data[10:, 0], evaluation_data[10:, 1], 'k')
    # plt.ylim(0.0001, 0.0015)
    # plt.ylim(3.0, 15.0)

    min_e_loss = np.min(evaluation_data[:, 1])
    min_e_ind = evaluation_data[np.argmin(evaluation_data[:, 1]), 0]
    min_t_loss = training_data[np.argmin(evaluation_data[:, 1]) * 10, 1]
    min_t_ind = min_e_ind
    #min_t_loss = np.min(training_data[:, 1])
    #min_t_ind = training_data[np.argmin(training_data[:, 1]), 0]
    print('Minimum eval loss: {}, at steps: {}'.format(min_e_loss, min_e_ind))
    print('Minimum train loss: {}, at steps: {}'.format(min_t_loss, min_t_ind))
    plt.plot(min_e_ind, min_e_loss, 'rx')
    plt.plot(min_t_ind, min_t_loss, 'rx')

    plt.ylabel("Loss")
    plt.title(
        model_name + '\n' + 'Training Loss: blue line   ' +
        ', Evaluation Loss: black line')
    P.text(0.48,0.875,'-', ha='center', va='bottom', size=24,color='blue')
    P.text(0.87,0.875,'-', ha='center', va='bottom', size=24,color='black')
    p2 = plt.subplot(212)
    #plt.plot(validation_data[:, 0], validation_data[:, 1], 'k')
    # plt.plot(training_data[100:, 0], training_data[100:, 3], 'b')
    # plt.plot(evaluation_data[10:, 0], evaluation_data[10:, 3], 'g')
    # plt.ylim(top=200)

    # min_e_error = np.min(evaluation_data[:, 3])
    # min_e_ind_err = evaluation_data[np.argmin(evaluation_data[:, 3]), 0]
    # min_t_error = training_data[np.argmin(evaluation_data[:, 3]) * 10, 3]
    # min_t_ind_err = min_e_ind_err
    # print('Minimum eval error: {}, at steps: {}'.format(min_e_error, min_e_ind_err))
    # print('Minimum train error: {}, at steps: {}'.format(min_t_error, min_t_ind_err))
    # plt.plot(min_e_ind_err, min_e_error, 'rx')
    # plt.plot(min_t_ind_err, min_t_error, 'rx')

    plt.xlabel("Steps")
    plt.ylabel("Evaluation Error")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()