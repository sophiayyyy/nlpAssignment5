#Assignment5

In this last assignment, the goal is to do a real-world NLP task (from SemEval: Semantic Evaluation workshop), using any tools or knowledge you have learned throughout the semester. 
The goals of this assignment are:
(1)	To predict, given a tweet in English, its most likely associated emoji
(2)	To predict, given a tweet in Spanish, its most likely associated emoji
(3)	To use any type of multilingual transfer learning to see if you can use English data to improve Spanish emoji prediction or vice versa (20 points)

1.run python3 main.py to get train and test tsv file for both Spanish and English data(for Spanish data just run main.py and for English data you should comment line6,11,37 and uncomment 7,12,38),move your English data(train and test) to data folder and Spanish data to es_data folder

2.for the first question you can run a5_en_train.sh for training and a5_en_pre.sh for prediction(for prediction part you should change the number of TRAINED_CLASSIFIER with highest checkpoint number you saw)

3.for the second question you can run a5_es_train.sh for training and a5_es_pre.sh for prediction(for prediction part you should change the number of TRAINED_CLASSIFIER with highest checkpoint number you saw)

4.for the last question, I use muse to do the translation. And it takes a really long time even on scc. So for the you can run test.py to get the first ten thousands data from English dataset to Spanish dataset. For all of them you can just change the number in 51 and 115 to have large dataset. (Up to 90000) I've already done the first 10k which can be seen in extend_data_test.csv
After run test.py you will get train_new.tsv, simply copy it to train folder and change it name into train.tsv and you will be ready for the training. (This time using a5_es_train_extend.sh and a5_es_pre_extend.sh)

5.after the prediction, you can get test_result.tsv(uncomment the file name first depend on which data set you want to convert) in your bert_output(or bert_output_es) folder. And then run python3 transresult.py to convert data into what evaluation python file want. (Pay attention to the row number, the last line will be missing with no reason after Bert prediction so you have to add one line to the result_en.txt or result_es.txt!!!) and then run python3 scorer_semeval18.py data/english_test.text result_en.txt to get the matrix and F-score.
