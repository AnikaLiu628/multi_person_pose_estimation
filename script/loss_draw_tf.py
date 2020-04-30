import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

#MPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_vsp_1.log
def main():
    model_name = 'MPPE_MOBILENET_V1_1.0_MSE_COCO_360_640_v4'
    path_root = '../logs/{}.log'.format(model_name)
    print('Model: ', model_name)
    training_data = []
    validation_data = []
    evaluation_data_loss = []
    training_data_loss = []
    evaluation_data = []
    pretrained_model_steps = 10000
    line_count = 0
    with open(path_root, 'r') as f:
        for line in f:
            if 'Evaluation Loss' in line:
                line_count += 1
                row = line.split(',')
                acc = 0
                for i, col in enumerate(row):
                    if i == 5:
                        col = col.split('(')[0]
                    if 'Evaluation Loss' in col:
                        loss = float(col.split('=')[-1])
                    # elif 'Error' in col:
                    #     err = float(col.split('=')[-1])
                    elif 'Eval_hm_Loss' in col:
                        hm = float(col.split('=')[-1])
                    elif 'Eval_paf_loss' in col:
                        paf = float(col.split('=')[-1])
                    # elif 'Rate' in col:
                    #     lr = float(col.split('=')[-1])
                    elif 'Steps' in col:
                        steps = int(col.split('=')[-1])
                    elif 'Batch Size' in col:
                        bs = int(col.split('=')[-1])
                if bs != 128:
                    # evaluation_data.append([steps, loss, lr])
                    evaluation_data.append([steps, loss, hm, paf])
                    # evaluation_data_loss.append([steps, hm, paf])

                    line_count = 0
            elif 'Training Loss' in line:
                row = line.split(',')
                for i, col in enumerate(row):
                    if i == 4:
                        # print(col)
                        col = col.split('(')[0]
                    if 'Training Loss' in col:
                        loss = float(col.split('=')[-1])
                    if 'Train_hm_Loss' in col:
                        hm = float(col.split('=')[-1])
                    # elif 'Error' in col:
                    #     err = float(col.split('=')[-1])
                    # elif 'Intensities' in col:
                    #     pass
                    elif 'Train_paf_Loss' in col:
                        paf = float(col.split('=')[-1])
                    elif 'Steps' in col:
                        steps = int(col.split('=')[-1])
                    # elif 'Rate' in col:
                    #     lr = float(col.split('=')[-1])
                # training_data.append([steps, loss, lr])
                training_data.append([steps, loss, hm, paf])
                # training_data_loss.append([steps, hm, paf])

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

    # plt.plot(training_data[100:, 0], training_data[100:, 1], 'b')
    # plt.plot(evaluation_data[0:, 0], evaluation_data[10:, 1], 'k')

    plt.plot(training_data[10:, 0], training_data[10:, 1], 'b')
    plt.plot(evaluation_data[1:, 0], evaluation_data[1:, 1], 'k')
    plt.ylim(0, 220)
    # plt.ylim(3.0, 15.0)
    avg_e_loss = np.average(evaluation_data[:, 1])
    min_e_loss = np.min(evaluation_data[:, 1])
    avg = np.average(evaluation_data[10:, 1])
    plt.axhline(avg, color= 'r')
    min_e_ind = evaluation_data[np.argmin(evaluation_data[:, 1]), 0]
    # min_t_loss = training_data[np.argmin(evaluation_data[:, 1]) * 2, 1]
    # min_t_ind = min_e_ind#######
    #min_t_loss = np.min(training_data[:, 1])
    #min_t_ind = training_data[np.argmin(training_data[:, 1]), 0]
    print('Minimum eval loss: {}, at steps: {}'.format(min_e_loss, min_e_ind))
    # print('Minimum train loss: {}, at steps: {}'.format(min_t_loss, min_t_ind))
    # plt.plot(min_e_ind, min_e_loss, 'rx')
    # plt.plot(min_t_ind, min_t_loss, 'rx')

    plt.ylabel("Loss")
    plt.title(
        model_name + '\n' + 'Training Loss: blue line   ' +
        ', Evaluation Loss: black line')
    P.text(0.48,0.875,'-', ha='center', va='bottom', size=24,color='blue')
    P.text(0.87,0.875,'-', ha='center', va='bottom', size=24,color='black')
    # P.text(0.48,0.875,'-', ha='center', va='bottom', size=24,color='black')
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
    

    PP = plt.figure(2)
    plt.plot(training_data[10:, 0], training_data[10:, 2], 'b')
    plt.plot(evaluation_data[1:, 0], evaluation_data[1:, 2], 'k')
    plt.ylim(0, 0.035)
    plt.title(model_name + '\n' + 'Training hm Loss: blue line   ' + ', Evaluation Loss: black line')
    PPP = plt.figure(3)
    plt.plot(training_data[10:, 0], training_data[10:, 3], 'b')
    plt.plot(evaluation_data[1:, 0], evaluation_data[1:, 3], 'k')
    plt.ylim(0, 45)
    plt.title(model_name + '\n' + 'Training paf Loss: blue line   ' + ', Evaluation Loss: black line')

    plt.show()

if __name__ == '__main__':
    main()