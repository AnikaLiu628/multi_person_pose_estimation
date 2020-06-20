import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

#MPE_SHUFFLENET_V2_1.0_MSE_COCO_360_640_vsp_1.log
parser = argparse.ArgumentParser(description='Plot Loss')
parser.add_argument('--model', type=str, default='MPPE_MOBILENET_THIN_0.75_MSE_COCO_368_432_v19')
args = parser.parse_args()

t_num = 1
v_num = t_num //10

loss_ylim = 0.1
hm_ylim = 0.02
paf_ylim = 0.08

val_batch_size = 64
def main():
    model_name = args.model
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
                    elif 'Eval_hm_Loss' in col:
                        hm = float(col.split('=')[-1])
                    elif 'Eval_paf_loss' in col:
                        paf = float(col.split('=')[-1])
                    elif 'Steps' in col:
                        steps = int(col.split('=')[-1])
                    elif 'Batch Size' in col:
                        bs = int(col.split('=')[-1])
                if bs != val_batch_size:
                    # evaluation_data.append([steps, loss])
                    evaluation_data.append([steps, loss, hm, paf])

                    line_count = 0
            elif 'Training Loss' in line:
                row = line.split(',')
                for i, col in enumerate(row):
                    if i == 4:
                        col = col.split('(')[0]
                    if 'Training Loss' in col:
                        loss = float(col.split('=')[-1])
                    if 'Train_hm_Loss' in col:
                        hm = float(col.split('=')[-1])
                    elif 'Train_paf_Loss' in col:
                        paf = float(col.split('=')[-1])
                    elif 'Steps' in col:
                        steps = int(col.split('=')[-1])
                # training_data.append([steps, loss])
                training_data.append([steps, loss, hm, paf])

    training_data = np.array(training_data)
    validation_data = np.array(validation_data)
    evaluation_data = np.array(evaluation_data)
    print('Training Data:', training_data.shape)
    print('Evaluation Data:', evaluation_data.shape)

    
    P = plt.figure(1)

    p1 = plt.subplot(211)
    # plt.plot(training_data[100:, 0], training_data[100:, 1], 'b')
    # plt.plot(evaluation_data[0:, 0], evaluation_data[10:, 1], 'k')

    plt.plot(training_data[t_num:, 0], training_data[t_num:, 1], 'b')
    plt.plot(evaluation_data[v_num:, 0], evaluation_data[v_num:, 1], 'k')
    plt.ylim(0, loss_ylim)
    # plt.ylim(3.0, 15.0)
    avg_e_loss = np.average(evaluation_data[:, 1])
    min_e_loss = np.min(evaluation_data[:, 1])
    avg = np.average(evaluation_data[v_num:, 1])
    # plt.axhline(avg, color= 'r')
    min_e_ind = evaluation_data[np.argmin(evaluation_data[:, 1]), 0]
    print('Minimum eval loss: {}, at steps: {}'.format(min_e_loss, min_e_ind))
    plt.ylabel("Loss")
    plt.title(
        model_name + '\n' + 'Training Loss: blue line   ' +
        ', Evaluation Loss: black line')
    P.text(0.48,0.875,'-', ha='center', va='bottom', size=24,color='blue')
    P.text(0.87,0.875,'-', ha='center', va='bottom', size=24,color='black')
    p2 = plt.subplot(212)

    plt.xlabel("Steps")
    plt.ylabel("Evaluation Error")
    plt.grid()
    P.savefig("../logs/{}.png".format(model_name))
    

    PP = plt.figure(2)
    plt.plot(training_data[t_num:, 0], training_data[t_num:, 2], 'b')
    plt.plot(evaluation_data[v_num:, 0], evaluation_data[v_num:, 2], 'k')
    plt.ylim(0, hm_ylim)
    plt.title(model_name + '\n' + 'Training hm Loss: blue line   ' + ', Evaluation Loss: black line')
    PP.savefig("../logs/{}_hm.png".format(model_name))
    PPP = plt.figure(3)
    plt.plot(training_data[t_num:, 0], training_data[t_num:, 3], 'b')
    plt.plot(evaluation_data[v_num:, 0], evaluation_data[v_num:, 3], 'k')
    plt.ylim(0, paf_ylim)
    plt.title(model_name + '\n' + 'Training paf Loss: blue line   ' + ', Evaluation Loss: black line')
    PPP.savefig("../logs/{}_vect.png".format(model_name))
    plt.show()

if __name__ == '__main__':
    main()