import torch

data_path = "data/p1ch4/jane-austen/1342-0.txt"
with open(data_path, encoding="utf8") as f:
    text = f.read()

lines = text.split("\n")
line = lines[200]
print(line)

ASCII_letter_count = 128
letter_t = torch.zeros(len(line), ASCII_letter_count)

for i, letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < ASCII_letter_count else 0
    letter_t[i][letter_index] = 1


def clean_words(input_str: str):
    punctuation = '.,;:"!?“”_-'
    word_list = input_str.lower().replace("\n", " ").split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list


words_in_line = clean_words(line)
print(line, words_in_line)

word_list = sorted(set(clean_words(text)))
word_to_index = {word: i for (i, word) in enumerate(word_list)}
print(len(word_to_index))

# onehot encoding
word_t = torch.zeros(len(words_in_line), len(word_to_index))
for i, word in enumerate(words_in_line):
    word_index = word_to_index[word]
    word_t[i][word_index] = 1
    print("{} {} {}".format(i, word_index, word))
