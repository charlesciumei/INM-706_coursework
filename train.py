# Modified from Lab 4 

import torch
from torch.nn import CrossEntropyLoss as CrossEntropyLoss
from torch.utils.data import DataLoader as DataLoader
from dataset import NovelSentencesData
from models import EncoderGRU, DecoderGRU
from attention_models import DecoderGRUWithAttention
from logger import Logger
from torch.utils.data import random_split
from utils import parse_arguments, read_settings


# Check if Apple Silicon GPU is available. Otherwise look for cuda device.
# Code adapted from Apple Silicon PyTorch developers page
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print (x)
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print ("MPS device not found.")

def train(model_settings, train_settings):
    dataset = NovelSentencesData(vocab_init=True, max_seq_len=model_settings['max_seq_length'])
    voc = dataset.vocab
    train_dataset_params = {'batch_size': train_settings['batch_size'], 'shuffle': True}
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 30% for training, adjust as needed - you should use 80%, leave it running it hyperion as it takes awhile
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Training DataLoader
    train_dataloader = DataLoader(train_dataset, **train_dataset_params)

    # GRU model seq2seq
    encoder = EncoderGRU(vocab_size=len(voc), embedding_size=model_settings['embedding_dim'], hidden_size=model_settings['hidden_dim'])
    if model_settings['use_attention']:
        decoder = DecoderGRUWithAttention(vocab_size=len(voc), embedding_size=model_settings['hidden_dim'],
                             hidden_size=model_settings['hidden_dim'])
    else:
        decoder = DecoderGRU(vocab_size=len(voc), embedding_size=model_settings['embedding_dim'], hidden_size=model_settings['hidden_dim'])
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print(encoder)
    print(decoder)
    encoder.train()
    decoder.train()

    #Optimizer
    optim_pars = {'lr': 1e-3, 'weight_decay': 1e-3}
    decoder_learning_ratio = 5.0

    total_epochs = train_settings['epochs'] # set training epochs to value in config yaml file
    if train_settings['one_optimizer']:
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), **optim_pars)
    else:
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=optim_pars['lr'])
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=optim_pars['lr'] * decoder_learning_ratio)
    torch.autograd.set_detect_anomaly(True)


    # X: one data point from the dataset, dimensions (1, num_seq, seq_len)
    # num_seq serves as a 'batch size': the number of phrases that varies from dialogue to dialogue
    # processed in one 'go' by the EncoderRNN
    process_before_update = 1 # originally set to 100
    current_batch = 0
    total_phrase_pairs = 0
    loss_function = CrossEntropyLoss()
    print("A total of {:d} batches in the dataset".format(len(train_dataloader)))
    total_epoch_loss = []
    print(len(train_dataloader))
    for e in range(total_epochs):
        epoch_loss = 0
        processed_total_batches = 0
        for id, (idx, (key, reply)) in enumerate(train_dataloader):
            # reset the optimizer if this is the first batch
            if not current_batch:
                total_phrase_pairs = 0
                loss_batch = 0
            if key.size()[1] == 0:
                continue
            if train_settings['one_optimizer']:
                optimizer.zero_grad()
            else:
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

            current_batch += 1
            key = key.to(device)
            reply = reply.to(device)
            batch_size = key.size()[1]
            # number of tokens in a sequence
            seq_length = key.size()[2]
            total_phrase_pairs += batch_size
            # reshape/transpose the
            # put sequence first
            data = key
            data = data.view(-1, batch_size, seq_length)[0]

            output_encoder, hidden_encoder = encoder(data)
            # output of the encoder RNN: (seq_len, num_seq, hidden) and (1, num_seq, hidden)
            # hidden is the hidden state of the last layer of the encoder RNN, the same as the
            # last state of the output, if the number of layers and directions == 1
            # input in the decoder is a sequence of indices of tokens starting with <S>
            data_response = reply
            # data_reponse dimensions: (1, batch_size, seq_length), the first token is start token
            # This is the input to the first cell of the decoder
            decoder_input = torch.LongTensor([[dataset.vocab.index(dataset.start_token) for _ in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # output encoder and decoder
            hidden_encoder = hidden_encoder[-1, :, :].unsqueeze(0)
            if model_settings['use_attention']:
                output_decoder, hidden_decoder = decoder(decoder_input, hidden_encoder, output_encoder)
            else:
                output_decoder, hidden_decoder = decoder(decoder_input, hidden_encoder)

            # output_decoder, hidden_decoder = seq2seq(data, reply)
            # get the top prediction, its index is the index of the vocabulary
            # dimensions (1, batch_size)
            top_predict = output_decoder.topk(1).indices
            targets = data_response[0, :, 1]
            targets = targets.to(device)
            # first loss
            loss = loss_function(output_decoder.squeeze_(0), targets)
            loss_dialogue = 0
            loss_dialogue += loss
            # loop on the remaining tokens, except the last one
            # outputs of the decoder and targets are for
            # each token in every phrase in the dialogue

            for idx in range(1, data_response.size()[2] - 1):
                decoder_input = top_predict.view(-1, batch_size)

                if model_settings['use_attention']:
                    output_decoder, hidden_decoder = decoder(decoder_input, hidden_encoder, output_encoder)
                else:
                    output_decoder, hidden_decoder = decoder(decoder_input, hidden_encoder)
                top_predict = output_decoder.topk(1).indices
                targets = data_response[0, :, idx + 1]
                # loss from the predicted vs true tokens
                loss = loss_function(output_decoder.squeeze_(0), targets)
                loss_dialogue += loss
            # normalize by the number of max_sequence_length, NOT batch_size!!!
            loss_dialogue = loss_dialogue / data_response.size()[2]
            # add dialogue loss to the batch loss
            loss_batch += loss_dialogue

            if not current_batch % process_before_update:
                current_batch = 0
                loss_batch = loss_batch / process_before_update
                epoch_loss += loss_batch.item()
                processed_total_batches += 1
                # print('Loss={0:.6f}, total phrase pairs in the batch = {1:d}'.format(loss_batch, total_phrase_pairs))
                loss_batch.backward()
                if train_settings['one_optimizer']:
                    optimizer.step()
                else:
                    encoder_optimizer.step()
                    decoder_optimizer.step()

        epoch_loss = epoch_loss / processed_total_batches
        logger.log({"epoch_loss": epoch_loss})
        print(
            'Loss={0:.6f}, total phrase pairs in the batch = {1:d}, total batches processed = {2:d}'.format(epoch_loss,
                                                                                                            total_phrase_pairs,
                                                                                                            processed_total_batches))
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            # 'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
        }, 'model_and_optimizer.pth')


if __name__ == '__main__':
    args = parse_arguments()

    # Read settings from the YAML file
    settings = read_settings(args.config)

    # Access and use the settings as needed
    model_settings = settings.get('model', {})
    train_settings = settings.get('train', {})
    print(model_settings)
    train(model_settings, train_settings)

