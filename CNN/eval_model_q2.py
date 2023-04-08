from __future__ import unicode_literals, print_function, division
from io import open
import matplotlib.pyplot as plt
import glob
import os
import unicodedata as ud
import string
import torch
import torch.nn as nn
import random
import time
import math

all_chars = string.ascii_letters + " .,;'-"
n_letters = len(all_chars) + 1  # Plus EOS marker
category_lines = dict()
all_categories = list()
criterion = nn.NLLLoss()
learning_rate = 0.0005
max_length = 20
epochs = 100000
print_every = 5000
plot_every = 500
loss_list = []

accuracy_list = []
total_accuracy = 0


def find_files(path):
    return glob.glob(path)


def char_to_ascii(s):
    return ''.join(
        c for c in ud.normalize('NFD', s)
        if ud.category(c) != 'Mn'
        and c in all_chars
    )


def read_lines(file):
    with open(file, encoding='utf-8') as some_file:
        return [char_to_ascii(line.strip()) for line in some_file]


for file in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(file))[0]
    all_categories.append(category)
    lines = read_lines(file)
    category_lines[category] = lines

num_categories = len(all_categories)
if num_categories == 0:
    raise RuntimeError('Data not found.')


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.input_to_hidden = nn.Linear(num_categories + input_size + hidden_size, hidden_size)
        self.input_to_output = nn.Linear(num_categories + input_size + hidden_size, output_size)
        self.output_to_output = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category_vec, input_vec, hidden):
        input_combined = torch.cat((category_vec, input_vec, hidden), 1)
        hidden = self.input_to_hidden(input_combined)
        output = self.input_to_output(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.output_to_output(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# Transforming category name to one hot vector
def category_to_one_hot(category_name):
    category_index = all_categories.index(category_name)
    category_one_hot = torch.zeros(1, num_categories)
    category_one_hot[0][category_index] = 1
    return category_one_hot


# Transforming name to one hot vector
def name_to_one_hot(line):
    categories = torch.zeros(len(line), 1, n_letters)
    for letter_index in range(len(line)):
        letter = line[letter_index]
        categories[letter_index][0][all_chars.find(letter)] = 1
    return categories


# Transforming ground truth to one hot vector
def target_tensor(line):
    letter_indexes = [all_chars.find(line[i]) for i in range(1, len(line))]
    letter_indexes.append(n_letters - 1)
    return torch.LongTensor(letter_indexes)


# Sample from a category and starting letter
def evaluate_model_q2(category_name, letter='A'):
    rnn_eval = RNN(n_letters, 128, n_letters)
    rnn_eval.load_state_dict(torch.load('rnn.pkl'))
    with torch.no_grad():  # no need to track history in sampling
        category_one_hot = category_to_one_hot(category_name)
        input = name_to_one_hot(letter)
        hidden = rnn_eval.init_hidden()
        output_name = letter
        for i in range(max_length):
            output, hidden = rnn_eval(category_one_hot, input[0], hidden)
            top_values, top_indices = output.topk(1)
            top_indices = top_indices[0][0]
            if top_indices == n_letters - 1:
                break
            else:
                letter = all_chars[top_indices]
                output_name += letter
            input = name_to_one_hot(letter)
        return output_name


# Get multiple names from one category and multiple starting letters
def generate_names(category_name, start_letters='ABC'):
    for start_letter in start_letters:
        print(evaluate_model_q2(category_name, start_letter))


if __name__ == '__main__':
    # train_model_q2()
    # print('Russian names:')
    # generate_names('Russian', 'RUS')
    # generate_names('Russian', 'RUS')
    # print('German names:')
    # generate_names('German', 'GER')
    # print('Spanish names:')
    # generate_names('Spanish', 'SPA')
    # print('Chinese names:')
    # generate_names('Chinese', 'CHI')
    print('portuguese names:')
    generate_names('Russian', 'ABCDEF')

