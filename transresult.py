import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__ == "__main__":
    # Load the result data
    with open('test/english_test.labels') as f:
       golden_label = [line.strip() for line in f]
    print(golden_label)

    # with open('test/spanish_test.labels') as f:
    #     golden_label = [line.strip() for line in f]
    # print(golden_label)

    with open('test_results.tsv', 'r') as tsv:
        train_result = [line.strip().split('\t') for line in tsv]

    print(len(train_result))

    final_result = []
    for index in range(len(train_result)):
        final_result.append(train_result[index].index(max(train_result[index])))

    with open("result_en.txt", "w") as out:
       for index in range(len(final_result)):
           out.write(str(final_result[index]))
           out.write("\n")

    # with open("result_es.txt", "w") as out:
    #     for index in range(len(final_result)):
    #         out.write(str(final_result[index]))
    #         out.write("\n")

    print("Now run: python scorer_semeval18.py results.txt")

    n_row = 20
    all_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_row, n_row)

    # Go through a bunch of examples and record which are correctly guessed
    val_correct_sum = 0.0
    macro_val = 0.0
    micro_val = 0.0
    tmp = 0.0
    val_correct = torch.zeros(n_row)
    val_incorrect = torch.zeros(n_row)

    num_confusion = 0
    for index in range(len(final_result)):
        i = all_label.index(str(final_result[index]))
        j = all_label.index(str(golden_label[index]))
        confusion[i][j] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_row):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_label, rotation=90)
    ax.set_yticklabels([''] + all_label)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()
    evaname = "evaluate.png"
    plt.savefig(evaname)






