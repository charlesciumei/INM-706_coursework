
import os
import re # regex operations
import torch
import numpy as np
import pickle
import random


from torch.utils.data import Dataset
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import punkt  # Not sure I need this
nltk.download('punkt')   # from nltk documentation in case it is not already downloaded


class NovelSentencesData(Dataset):
    UNK = '<UNK>'
    START_TOKEN = "<BOS>"
    END_TOKEN = "<EOS>"

    def __init__(self, max_seq_len=20, vocab_init=False):
        self.data = self._pre_process_text(os.path.join('dataset', 'Can_You_Forgive_Her_Chs_1_10.txt'))
        self.max_seq_len = max_seq_len # default is 20 words because Trollope uses long sentences
        self.unk_token = self.UNK
        self.end_token = self.END_TOKEN
        self.start_token = self.START_TOKEN

        if vocab_init:
            self.vocab = self.build_vocab()
            with open("vocabulary1.pkl", "wb") as file:
                pickle.dump(self.vocab, file)
        else:
            self.load_vocab()
        
        # print(self.data)
        # print(self.vocab) # remove this later. just to check I am getting a vocab

# reads in text from novel and returns a list of sentences without punctuation
# but with <BOS> and <EOS> tokens

# then dataloader will load contiguous pairs of sentences as prompt:response

    def _pre_process_text(self, filepath):
        with open (filepath, 'r', encoding="utf-8") as f:
            text = f.read()
        text = re.sub('\n', ' ', text) # get rid of new lines
        text = text.lower() # make everything lowercase
        text = sent_tokenize(text) # break text into sentences
        # sentence_tok_text = [word_tokenize(word) for word in sentence_tok_text]
        # hangover from old version. NB nltk use of expression tokenize does NOT
        # mean tokenise by changing words to numbers
        sentence_tok_text =[]
        for sentence in text:
            sentence = f"<BOS> {sentence} <EOS>" # add <BOS> <EOS> tokens with a space 
            sentence = re.sub(r'[^\w<>]+', ' ', sentence) # regex to remove punctuation except token markers
            sentence_tok_text.append(sentence)
        return sentence_tok_text # return list of sentences

    # Build the vocabulary, use sorted to ensure the same order
    def build_vocab(self):
        text = self.data
        vocab = []
        excluded_items = ['<', '>', 'BOS', 'EOS'] # add them back at the end otherwise tokenizer separates them

        for sentence in text:
            for word in word_tokenize(sentence):
                if word in excluded_items:
                    pass
                else:
                    if word not in vocab:
                        vocab.append(word)
        vocab.append(self.end_token)
        vocab.append(self.start_token)
        vocab.append(self.unk_token)
        print('Vocabulary size: ', len(vocab))
        return sorted(set(vocab)) # return the vocab_list sorted    

    # load an existing vocabulary
    def load_vocab(self):
        path_vocab = os.path.join(os.getcwd(), 'vocabulary1.pkl')
        if os.path.exists(path_vocab):
            # Load the object back from the file
            with open(path_vocab, "rb") as file:
                self.vocab = pickle.load(file)
        print('Vocabulary size: ', len(self.vocab))
        return


# then convert the sentences to numbers (properly tokenize)
# then convert the number-sentences to torch tensors

    # makes sentences uniform length and pads with <UNK> as if necessary (taken from Lab 4 code)
    def trim_sentence(self, tokenized_k):
        if len(tokenized_k) > self.max_seq_len:
            tokenized_k = tokenized_k[:self.max_seq_len]
            tokenized_k.append(self.end_token)
        elif len(tokenized_k) <= self.max_seq_len:
            tokenized_k.append(self.end_token)
            seq_pad = (self.max_seq_len - len(tokenized_k) + 1) * [self.unk_token]
            if len(seq_pad) > 0:
                tokenized_k.extend(seq_pad)
        return tokenized_k

    # create a dialogue (effectively a batch) of contiguous sentences 
    def load_sentences(self):
        dialogue_dict = {}
        upper_limit = len(self) - 10  # prevent sample range beyond size of dataset
        random_sample_start = random.randint(0, upper_limit)
        for value in range(random_sample_start, random_sample_start+9):
            dialogue_dict[self.data[value]] = self.data[value+1]
        # print(dialogue_dict)
        return dialogue_dict   
        
    
    # loads one full dialogue (K phrases in a dialogue), an OrderedDict
    # If there are a total of K phrases, the data point will be with dimensions
    # ((K-1) x (MAX_SEQ + 2), (K-1) x (MAX_SEQ + 2))
    # zero-pad after EOS
    def load_dialogue(self):
        dialogue_dict = self.load_sentences()
        try:
            # k: phrase
            all_inputs = []
            all_outputs = []
            # get keys (first phrase) from the dialogues
            keys = dialogue_dict.keys()
            # print(keys)
            for k in keys:
                # print(k)
                # tokenize here, both key and reply
                tokenized_k = re.sub(r'[^\w<>]+', ' ', k).split()
                # print(f'this is the tokenized key {tokenized_k}')
                tokenized_r = re.sub(r'[^\w<>]+', ' ', dialogue_dict[k]).split()

                # pad or truncate, both key and reply
                tokenized_k = self.trim_sentence(tokenized_k)
                # print(tokenized_k)
                tokenized_r = self.trim_sentence(tokenized_r)

                # Convert to indices - query
                input_phrase = [self.vocab.index(w) for w in tokenized_k]
                # print(input_phrase)
                input_phrase.insert(0, self.vocab.index(self.start_token))

                output_phrase = [self.vocab.index(w) for w in tokenized_r]
                output_phrase.insert(0, self.vocab.index(self.start_token))

                # append to the inputs and outputs - queries and replies
                all_inputs.append(torch.tensor(input_phrase))
                all_outputs.append(torch.tensor(output_phrase))
            all_inputs = torch.stack(all_inputs)
            all_outputs = torch.stack(all_outputs)
            # return a tuple
            output_tuple = (all_inputs, all_outputs)

            return output_tuple
        except Exception as e:  # report the error
            print(f"An error occurred: {e}")
            return (torch.tensor([]), torch.tensor([]))




    # number of sentences in the dataset
    def __len__(self):
        return len(self.data)
 
    # returns a tuple of 2 torch tensors 
    def __getitem__(self, idx):      # this needs to be developed to 2 sentences
        # self.dialogue = self.movies_data.iloc[idx]
        self.phrases = self.load_dialogue()
        return idx, self.phrases
        # return self.data[idx]

# script to run when the file is called on its own and not as a module
def run():
    foo = NovelSentencesData()
    print(foo[0])
    print(len(foo))

if __name__ == '__main__':
    run()

