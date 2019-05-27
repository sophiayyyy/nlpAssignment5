import csv
import pandas as pd

if __name__ == "__main__":
    # Load the training data
    with open('train/spanish_train.labels') as f:
    #with open('train/english_train.labels') as f:
        train_label = [line.strip() for line in f]

    print('length of train_label', len(train_label))
    with open('train/spanish_train.text') as f:
    #with open('train/english_train.text') as f:
        train_text = [line.strip() for line in f]

    print('length of train_label', len(train_text))

    # process train data
    train_data = []
    for index in range(len(train_label)):
        line = [int(train_label[index]), 'a', train_text[index]]
        train_data.append(line)

    print(len(train_data))

    with open('train.csv', mode='w') as f:
        file_writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for index in range(len(train_data)):
            file_writer.writerow(train_data[index])

    #convert csv to tsv
    # if you are creating test.tsv, set header=True instead of False
    df_train = pd.read_csv('train.csv', sep='\t', header=None)
    print(df_train)
    df_train.to_csv('train.tsv', sep='\t', header=0)

    # Load the test data
    with open('test/spanish_test.text') as f:
    #with open('test/english_test.text') as f:
        test_text = [line.strip() for line in f]
    print('length of test', len(test_text))

    # process test data
    test_data = []
    for index in range(len(test_text)):
        line = [train_text[index]]
        test_data.append(line)
    with open('test.csv', mode='w') as f:
        file_writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for index in range(len(test_data)):
            file_writer.writerow(test_data[index])

    # convert csv to tsv
    # if you are creating test.tsv, set header=True instead of False
    df_test = pd.read_csv('test.csv', sep='\t', header=None, error_bad_lines=False)
    print(df_test)
    df_test.to_csv('test.tsv', sep='\t', header=0)




