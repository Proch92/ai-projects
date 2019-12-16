from lstm import LSTM
import datautils
import utils
import sys
import numpy as np


def main():
    argv = sys.argv

    if len(argv) != 4:
        print('Usage: ' + argv[0] + ' model_name dataset original_dataset')
        sys.exit(0)

    model_name = argv[1]
    data = datautils.load(argv[2])
    original_data = datautils.load(argv[3])
    normalized, mean, std = datautils.normalize(data)

    (eval_sequences, cuts_indexes) = split_evaluation_sequences(normalized)

    """eval"""
    model = LSTM()

    clean = np.empty(0)
    for head, tail in eval_sequences:
        if len(clean) == 0:
            y_init = 0
        else:
            y_init = clean[-1]
        head_diff = datautils.differentiate(head, y_init)

        projection = model.evaluate(model_name, head_diff, tail)

        head = datautils.undifferentiate(head_diff, y_init)
        projection = datautils.undifferentiate(projection, head[-1])

        clean = np.concatenate((clean, head, projection))

    """plot"""
    clean_denorm = datautils.denormalize(clean, mean, std)
    utils.plot_multiple(
        [original_data, clean_denorm],
        [0, 0],
        vertical_lines=cuts_indexes)


#############################################################
# takes a sequence with empty values and returns a list     #
# of objects in the form (sub sequence, projection length)  #
# input:                                                    #
# [1,2,3,4,5,,,,3,4,5,6,,,2,3,4,5]                          #
# output:                                                   #
# [([1,2,3,4,5], 3),                                        #
# ([3,4,5,6], 2)]                                           #
# also returns indexes of cuts for plotting                 #
#############################################################

def split_evaluation_sequences(data):
    boolean_sequences = np.isnan(data)
    state = boolean_sequences[0]
    cuts_indexes = []
    cuts = []
    seq = []
    nan_counter = 0
    for i, nan in enumerate(boolean_sequences):
        # state change
        if nan != state:
            cuts_indexes.append(i)
            if state is True:
                print("cut: index {}, length {}".format(i, nan_counter))
                cuts.append((seq, nan_counter))
                seq = []
                nan_counter = 0

        if nan:
            nan_counter = nan_counter + 1
        else:
            seq.append(data[i])

        state = nan

    return (cuts, cuts_indexes)


if __name__ == '__main__':
    main()
