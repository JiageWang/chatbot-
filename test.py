import torch.nn as nn
from model import EncoderRNN, DecoderRNN
from utils import *

device = 'cpu'

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_TOKEN
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_WORDS):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [sentence2indexes(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evaluateInput(encoder, decoder, searcher, voc, language='en'):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')

            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break

            # Normalize sentence
            if language == 'en':
                input_sentence = normalizeString(input_sentence)
            elif language == 'ch':
                input_sentence = normalizeInputChinese(input_sentence)

            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)

            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            if language == 'en':
                print('Bot:', ' '.join(output_words))
            elif language == 'ch':
                print('Bot:', ''.join(output_words))


        except KeyError as e:
            print(e)
            print("Error: Encountered unknown word.")

if __name__ == '__main__':
    state_path = r'model/chatbot/xiaohuangji/2-2_800/6000_checkpoint.tar'
    voc_pkl = 'pkl/xiaohuangji_voc.pkl'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    voc = torch.load(voc_pkl)
    embedding = nn.Embedding(voc.num_words, hidden_size)
    state = torch.load(state_path.format(dataset_name))
    embedding.load_state_dict(state['embedding'])

    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layer, dropout)
    decoder = DecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layer, dropout)
    encoder.load_state_dict(state['en'])
    decoder.load_state_dict(state['de'])


    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    searcher = GreedySearchDecoder(encoder, decoder)

    language = 'ch' if dataset_name=='xiaohuangji' else 'en'
    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(encoder, decoder, searcher, voc, language)
