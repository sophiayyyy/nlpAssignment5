import csv
import pandas as pd
import io
import numpy as np

def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    #print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    if word not in word2id:
        #print('not exist')
        return ' '
    else:
        word_emb = src_emb[word2id[word]]
        scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
        k_best = scores.argsort()[-K:][::-1]
        for i, idx in enumerate(k_best):
            #print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
            return tgt_id2word[idx]

if __name__ == "__main__":
    # Load the training data
    # with open('train/english_train.text') as f:
    # #with open('train/english_train.text') as f:
    #     train_text = [line.strip() for line in f]
    # print('length of train_label', len(train_text))
    #
    # all_word = []
    # for line in train_text:
    #     all_word.append(line.split(' '))
    #
    # print(len(all_word))
    #
    # test_word =[]
    # for index in range(10000):
    #     test_word.append(all_word[index])
    #
    # src_path = 'wiki.multi.en.vec'
    # tgt_path = 'wiki.multi.es.vec'
    # nmax = 50000  # maximum number of word embeddings to load
    #
    # src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax)
    # tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path, nmax)
    #
    # for i in range(len(test_word)):#each line test_word[i]
    #     print('row:', i)
    #     for j in range(len(test_word[i])):#each word in line  test_word[i][j]
    #         #check illegal character
    #         for char in test_word[i][j]:
    #             if not char.isalpha():
    #                 test_word[i][j] = test_word[i][j].replace(char, "")
    #
    #         if test_word[i][j]:
    #             test_word[i][j]  = get_nn(test_word[i][j].lower(), src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, K=1)
    #
    # # convert
    # str = ' '
    # for index in range(len(test_word)):
    #     test_word[index] = str.join(test_word[index])
    # print(test_word)
    # # write new tsv data file
    # with open('extend_data_test.csv', mode='w') as f:
    #     for line in test_word:
    #         f.write(line)
    #         f.write('\n')



    # load all spanish data
    text = []
    with open('extend_data_test.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            text.append(row[0])
    print('length of extend_data_test', len(text))
    csvFile.close()

    with open('train/spanish_train.text') as f:
    #with open('train/english_train.text') as f:
        train_text = [line.strip() for line in f]
    print('length of train_text', len(train_text))

    final_text = text + train_text
    print('length of final_text', len(final_text))

    # load all spanish label
    with open('train/spanish_train.labels') as f:
        # with open('train/english_train.labels') as f:
        spainsh_label = [line.strip() for line in f]

    print('length of spanish_train', len(spainsh_label))

    with open('train/english_train.labels') as f:
        # with open('train/english_train.labels') as f:
        english_label = [line.strip() for line in f]

    print('length of english_train', len(english_label))

    for i in range(len(english_label)):
        if english_label[i] == 4:
            english_label[i] = 19
        elif english_label[i] == 5:
            english_label[i] = 4
        elif english_label[i] == 6:
            english_label[i] = 10
        elif english_label[i] == 7:
            english_label[i] = 15
        elif english_label[i] == 8:
            english_label[i] = 11
        elif english_label[i] == 9:
            english_label[i] = 5
        elif english_label[i] == 10:
            english_label[i] = 19
        elif english_label[i] == 11:
            english_label[i] = 19
        elif english_label[i] == 12:
            english_label[i] = 19
        elif english_label[i] == 13:
            english_label[i] = 12
        elif english_label[i] == 14:
            english_label[i] = 7
        elif english_label[i] == 15:
            english_label[i] = 19
        elif english_label[i] == 16:
            english_label[i] = 18
        elif english_label[i] == 17:
            english_label[i] = 19
        elif english_label[i] == 18:
            english_label[i] = 19
        elif english_label[i] == 19:
            english_label[i] = 13
        else:
            english_label[i] = english_label[i]

    final_label = spainsh_label + english_label[:10000]
    print('length of final_label', len(final_label))

    # process train data
    train_data = []
    for index in range(len(final_label)):
        line = [int(final_label[index]), 'a', final_text[index]]
        train_data.append(line)

    print('length of train_data', len(train_data))

    with open('train_new.csv', mode='w') as f:
        file_writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for index in range(len(train_data)):
            file_writer.writerow(train_data[index])

    # convert csv to tsv
    # if you are creating test.tsv, set header=True instead of False
    df_train = pd.read_csv('train_new.csv', sep='\t', header=None)
   # print(df_train)
    df_train.to_csv('train_new.tsv', sep='\t', header=0)










