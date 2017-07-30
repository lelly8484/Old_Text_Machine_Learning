# Old_Text_Machine_Learning/combined model

* Python scripts

1 predict.py : start all steps of frequency model, word-level model, and char-level model.
- Input : all xmls under XML directory.
2. freq_get_input.py : convert all xmls into combined text & extract the text excluding all special characters & extract list of words with dots.
3. freq_train.py : get the list of the prediction of dot words.
4. freq_get_output.py : give the output of modified xmls after correction.
5. word_get_input.py : convert all xmls into combined text.
- Input : the output of xmls from the frequency model.
6. word_preprocess.py : generate training data and testing data as numpy array of embedding.
7. word_train.py : train the word-level language model & give the list of prediction of dot words.
8. word_get_output.py : convert combined text into each xmls.
- word_lstm_model.py : definition the word-level language model.
- word_corpus_data.py : tokenize all the file contents.
9. char_get_input.py : convert all xmls into combined text.
- Input : the output of xmls from the word-level language model.
10. char_preprocess.py : generate training data and testing data as numpy array of embedding.
11. char_train.py : train the char-level language model & give the list of prediction of dot words.
12. char_get_output.py : convert combined text into each xmls.
- char_lstm_model.py : definition the char-level language model.

* Directories
1. XML : (INITIAL INPUT) all xml files
2. freq_data : text files and data file for the frequency model
3. freq_output : fixed xml files from the frequency model
4. word_data : text files and data file for the word-level model
5. word_ptb_models : word-level language model
6. word_output : fixed xml files from the word-level model
7. char_data : text files and data file for the char-level model
8. char_ptb_models : char-level language model
9. FIXED_XML : (FINAL OUTPUT) fixed xml files from the char-level model