from config import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN


class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: 'PAD', SOS_TOKEN: 'SOS', EOS_TOKEN: 'EOS'}
        self.num_words = 3

    def addSentence(self, sentence):
        '''添加语句中每个单词至单词表'''
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        '''添加单词至单词表'''
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        '''清除单词频次低的单词'''
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {}/{} = {:.4f}'.format(len(keep_words), len(self.word2index), len(keep_words ) /len(self.word2index)))
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: 'PAD', SOS_TOKEN: 'SOS', EOS_TOKEN: 'EOS'}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)



def trimRareWords(voc, pairs, MIN_COUNT):
    '''去除低频单词以及单词所在语句'''
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)
    print('Trimmed from {} pairs to {}, {:.4f} of total'.format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs