import os
import torch
import torch.optim as optim
import torch.nn as nn
import random
from words import Vocabulary, trimRareWords
from dataset import get_pairs
from model import EncoderRNN, DecoderRNN
from utils import *
from tensorboardX import SummaryWriter


def maskNLLLoss(decoder_out, target, mask):
    nTotal = mask.sum()
    target = target.view(-1,1)
    gathered_tensor = torch.gather(decoder_out, 1, target)
    crossEntropy = -torch.log(gathered_tensor)
    loss = crossEntropy.masked_select(mask)
    loss = loss.mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, ):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_TOKEN for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, dataset_name, summary):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder,  encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            summary.add_scalar('loss', print_loss_avg, iteration)
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, dataset_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

        if iteration in [1600, 4000, 8000]:
            for param_group in encoder_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay_rate
            for param_group in decoder_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay_rate

if __name__ == "__main__":
    summary = SummaryWriter('./summary/'+dataset_name+'_'+str(hidden_size))

    # 数据集
    pairs = get_pairs(dataset_name)

    # 单词表
    voc_path = './pkl/dataset_name'+'_voc.pkl'
    if os.path.exists(voc_path):
        exit(voc_path+' is already exists, please remove first')

    voc = Vocabulary(dataset_name)
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])

    # 去除低频单词
    pairs = trimRareWords(voc, pairs, MIN_COUNT)
    # 保存voc
    print('Saving voc...')
    torch.save(voc, voc_path)
    print('Done\n')

    print('Counted words:', voc.num_words)

    # 设备
    device = ['cpu', 'cuda'][int(torch.cuda.is_available())]
    print('Device:', device, '\n')

    # 模型
    embedding = nn.Embedding(voc.num_words, hidden_size)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layer, dropout)
    decoder = DecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layer, dropout)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.train()
    decoder.train()

    # 优化器
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate*decoder_learning_ratio)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 导入与训练模型
    if loadFilename:
        print('Loading state...')
        state = torch.load(loadFilename)
        encoder.load_state_dict(state['en'])
        decoder.load_state_dict(state['de'])
        embedding.load_state_dict(state['embedding'])
        print('Done\n')

    # 开始训练
    print("Starting Training!")
    trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, dataset_name, summary)
    print('Done\n')

