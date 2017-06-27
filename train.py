import argparse
import torch
#import torch.cuda as cuda
from torch.autograd import Variable
from model import *
import time
import math
import string
import pickle
import numpy as np
import bisect
import matplotlib.pyplot as plt


#################################################
# Hyper-parameters
#################################################

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank Character-Level LSTM Model')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--load_epochs', type=int, default=0,
                    help='load epoch')
parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='use Bi-LSTM')
parser.add_argument('--serialize', action='store_true', default=False, #False,
                    help='continue training a stored model')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
args = parser.parse_args()


#############################################
# Load data
#############################################
all_letters = string.ascii_letters+string.punctuation+string.digits+" ̄ÁáÀàÅåÄäāâÇçÉéÈèÊêëēÍíÌìÎîÑñÓóÒòÔôÖöōÚúÙùÜüūŽž  " #string.ascii_letters + " ,.;\'\\1234567890&*-/"
n_letters = len(all_letters)

all_categories = list(string.ascii_letters) #+string.punctuation)#+" ÁáÀàÅåÄäāÇçÉéÈèÊêÍíÌìÎîÑñÓóÒòÔôÖöÚúÙùÜüŽžv̄") #[:52])   # Using only English letters as categories
n_categories = len(all_categories)

category_lines = []

# Load dictionaries of actual asterisked characters
with open('./data/test_data', 'rb') as handle:
    test_data = pickle.load(handle)

# Load string of asterisked training and validation data
with open('./data/train_data_array', 'rb') as handle:
    train_data_array = pickle.load(handle)
with open('./data/val_data_array', 'rb') as handle:
    val_data_array = pickle.load(handle)

all_data_array = np.append(train_data_array, val_data_array)


########################################################
# Pre-process testing data
# Generate list of target index and tensor of target
########################################################
#for testing we had index and target, we want to split the letter into list of tensor and the numbers into list of indeces 
def char_index(ch):
    return all_letters.index(ch)

test_data.sort(key=lambda x: x[0], reverse=False)

test_target_index = []
test_target_tensor = torch.LongTensor(len(test_data)).zero_()

for i, char in enumerate(test_data):
    test_target_index.append(char[0])
    test_target_tensor[i] = char_index(char[1])


########################################################
# Pre-process training and validation data
########################################################
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[0: nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    data = data.reshape(bsz, -1)
    return data


# Wraps hidden states in new Variables, to detach them from their history.
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

# for every batch you call this to get the batch of data and targets, in testing your only getting astericks for the characters you took out, 
# you have the index but the data is turned into 3D array, you have to map indices that you have, 
#you have to search for what astericks/how many are in the array, and then you go back using the indices of the astericks you already know
# to map that into indices of output array that you have 
def get_batch(source, source_target, i, evaluation=False):
    seq_len = min(args.bptt, source.size(1) - 1 - i)  # -1 so that there's data for the target of the last time step
    data = Variable(source[:, i: i + seq_len], volatile=evaluation)   # Saves memory in purely inference mode
    if args.bidirectional:
        # r_target of length seq_len - 1
        r_source_target = np.flip(source_target[:, i - 1: i - 1 + seq_len].cpu().numpy(), 1).copy()
        target = torch.cat((Variable(source_target[:, i + 1: i + 1 + seq_len].contiguous().view(-1)),
                            Variable(torch.from_numpy(r_source_target).cuda().contiguous().view(-1))), 0)
    else:
        target = Variable(source_target[:, i + 1: i + 1 + seq_len].contiguous().view(-1))
    return data, target


def embed(data_array, bsz):
    # convert 1D array to 2D
    data_array = batchify(data_array, bsz)

    # initialize train_data_tensor, test_data_tensor
    data_tensor = torch.FloatTensor(data_array.shape[0], data_array.shape[1], n_letters).zero_()

    # convert 2D numpy array to np.int64
    data_array = data_array.astype(np.int64)

    # convert 2D numpy array to 3D tensor embedding
    for i in range(0, data_array.shape[0]):
        for j in range(0, data_array.shape[1]):
            data_tensor[i][j][data_array[i][j]] = 1

    # convert 2D numpy array to 2D target tensor
    target_tensor = torch.LongTensor(data_array)

    return data_tensor, target_tensor

val_bsz = 5
# convert list of indeces into tensors and does batching 
# lettesr were converted earlier
train_data_tensor, train_target_tensor = embed(train_data_array, args.batch_size)
val_data_tensor, val_target_tensor = embed(val_data_array, val_bsz)
all_data_tensor, all_target_tensor = embed(all_data_array, args.batch_size)


###############################################################################
# Helper functions for searching within a sorted list
###############################################################################
def find_ge(a, x):
    """Find leftmost item greater than or equal to x"""
    i = bisect.bisect_left(a, x)
    return i


def find_le(a, x):
    """Find rightmost value less than or equal to x"""
    i = bisect.bisect_right(a, x)
    return i - 1

###############################################################################
# Build the model
###############################################################################
if args.bidirectional:
    name = 'Bi-LSTM'
else:
    name = 'LSTM'

if args.serialize:
    with open('./PTB Models/{}_Epoch{}_BatchSize{}_Dropout{}_LR{}_HiddenDim{}.pt'.format(
             name, args.load_epochs, args.batch_size, args.dropout, args.lr, args.nhid), 'rb') as f:
        model = torch.load(f)
else:
    model = MyLSTM(n_letters, args.nhid, args.nlayers, True, True, args.dropout, args.bidirectional, args.batch_size, args.cuda)
    # input_dim, hidden_dim, category_dim, layers, bias (unsure), batch_first, dropout, bidirectional, batch_size, cuda
    args.load_epochs = 0

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax()
NLL = nn.NLLLoss()

if args.cuda:
    criterion.cuda()
    softmax.cuda()
    NLL.cuda()


def train():
    # Turn on training mode which enables dropout.
    # Built-in function, has effect on dropout and batchnorm
    model.train()
    total_loss = 0
    start_time = time.time()
    #####reinitialize hidden states, get a batch, runs every epoch, then train on the batches, then you calculate
    #### the loss and run backprop
    hidden = model.init_hidden(args.batch_size)
    # (0, total length of sequences in each batch row, sequence length)
    # for Bi-LSTM, leave out the first character in inputs for reverse loss calculation
    for batch, i in enumerate(range(1, train_data_tensor.size(1) - 1, args.bptt)):
        # returns Variables
        data, targets = get_batch(train_data_tensor, train_target_tensor, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # If bidirectional, reinitialize hidden state every batch
        if not args.bidirectional:
            hidden = model.init_hidden(args.batch_size)
        else:
            #backprop within own batch
            hidden = repackage_hidden(hidden)
        model.zero_grad()
        # remap input output
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)   # (scalar multiplier, other tensor)

        # torch.cuda.FloatTensor
        total_loss += loss.data
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:5.2f} |'.format(
                   epoch, batch, train_data_tensor.size(1) // args.bptt, lr,
                   elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Uses training data to generate predictions, calculate loss based on validation/testing data
# Not using bptt
def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(val_bsz)
    start_time = time.time()
    for batch, i in enumerate(range(1, val_data_tensor.size(1) - 1, args.bptt)):
        # returns Variables
        data, targets = get_batch(val_data_tensor, val_target_tensor, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        if not args.bidirectional:
            hidden = model.init_hidden(val_bsz)
        else:
            hidden = repackage_hidden(hidden)
        # remap input output
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        # torch.cuda.FloatTensor
        total_loss += loss.data
        if batch % (args.log_interval // 20) == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| validation | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  .format(batch, val_data_tensor.size(1) // args.bptt, lr,
                          elapsed * 1000 / (args.log_interval // 20)))
            start_time = time.time()
    return total_loss[0] / (val_data_tensor.size(1) // args.bptt)  # return loss per character


def test():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    correct = 0
    total_count = 0
    high_correct = 0
    high_total = 0
    last_forward = None
    batch_length = all_data_array.size // args.batch_size
    hidden = model.init_hidden(args.batch_size)
    start_time = time.time()
    for batch, i in enumerate(range(1, batch_length - 1, args.bptt)):
        # returns Variables
        data, _ = get_batch(all_data_tensor, all_target_tensor, i, evaluation=True)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        if not args.bidirectional:
            hidden = model.init_hidden(args.batch_size)
        else:
            hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        if args.bidirectional:
            # split output into forward and reverse chunks
            output_forward, output_reverse = torch.chunk(output, 2, 0)
        # Get loss and accuracy using val_target_index and val_target_tensor
        output_select = Variable(torch.FloatTensor(output.size(0), n_letters).zero_())
        target_select = Variable(torch.LongTensor(output.size(0)).zero_())
        count = 0
        bptt = min(batch_length - i, args.bptt)
        # batch_i: index of minibatch
        for batch_i in range(0, args.batch_size):
            if args.bidirectional:
                # find targets in the intersection of current forward and reverse output
                start = find_ge(test_target_index, i + 1 + batch_i * batch_length)
                end = find_le(test_target_index, i - 2 + batch_i * batch_length + bptt)
                for ii in range(start, end):
                    target = test_target_index[ii]
                    output_select[count] = torch.add(output_forward[target - batch_i * (batch_length - bptt) - i - 1],
                                                     output_reverse[-target + i + 1 + batch_i * (batch_length + bptt)])
                    target_select[count] = test_target_tensor[ii]
                    count += 1

                # find targets in the intersection of last forward and current reverse output
                if i != 1:
                    start = find_ge(test_target_index, i - 1 + batch_i * batch_length)
                    end = find_le(test_target_index, i + batch_i * batch_length)
                    for ii in range(start, end):
                        target = test_target_index[ii]
                        output_select[count] = torch.add(last_forward[target - batch_i * (batch_length - bptt) - (i - bptt) - 1],
                                                         output_reverse[-target + i + 1 + batch_i * (batch_length + bptt)])
                        target_select[count] = test_target_tensor[ii]
                        count += 1
                #store current forward output
                last_forward = output_forward
            else:
                start = find_ge(test_target_index, i + 1 + batch_i * batch_length)
                end = find_le(test_target_index, i + batch_i * batch_length + bptt) + 1
                for ii in range(start, end):
                    target = test_target_index[ii]
                    output_select[count] = output[target - batch_i * (batch_length - bptt) - i - 1]
                    target_select[count] = test_target_tensor[ii]
                    count += 1

        # store current forward output
        #last_forward = output_forward

        if count != 0:
            output_select = output_select[:count, :]
            target_select = target_select[:count]
            # calculate probability distribution over English letters
            output_prob = softmax(output_select[:, :n_categories])
            for n, target in enumerate(target_select.data):
                # Extract Tensor out of Variable with .data
                top_n, top_i = output_prob[n].data.topk(1)
                category_i = top_i.cpu().numpy()
                total_count += 1
                if category_i[0] == target:
                    correct += 1
                # check whether the probability of top choice > 0.9
                if top_n.cpu().numpy()[0] > 0.5:#0.9
                    high_total += 1
                    if category_i[0] == target:
                        high_correct += 1

            # use softmaxed result instead to speed up
            output_log = torch.log(output_prob)
            total_loss += NLL(output_log, target_select).data * count
            # total_loss += criterion(output_select, target_select).data * count

        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| test | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} | accuracy {:.2f} |'
                  .format(batch, batch_length // args.bptt, elapsed * 1000 / args.log_interval,
                          total_loss[0] / total_count, correct / total_count))
            start_time = time.time()
    print(total_loss[0] , total_count, correct , total_count, high_correct ,high_total, high_total , total_count)
    return total_loss[0] / total_count, correct / total_count, high_correct / high_total, high_total / total_count

# Loop over epochs.
lr = args.lr
best_val_loss = None

#At any point you can hit Ctrl + C to break out of training early.
try:
    if args.cuda:
        train_data_tensor = train_data_tensor.cuda()
        train_target_tensor = train_target_tensor.cuda()
        val_data_tensor = val_data_tensor.cuda()
        val_target_tensor = val_target_tensor.cuda()
    for epoch in range(args.load_epochs+1, args.epochs+args.load_epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} |'.format(
            epoch, (time.time() - epoch_start_time),
            val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open('./PTB Models/{}_Epoch{}_BatchSize{}_Dropout{}_LR{}_HiddenDim{}.pt'.format(
               name, args.load_epochs+args.epochs, args.batch_size, args.dropout, args.lr, args.nhid), 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
   print('-' * 89)
   print('Exiting from training early')

# Load the best saved model.
with open('./PTB Models/{}_Epoch{}_BatchSize{}_Dropout{}_LR{}_HiddenDim{}.pt'.format(
               name, args.load_epochs+args.epochs, args.batch_size, args.dropout, args.lr, args.nhid), 'rb') as f:
    model = torch.load(f)

# Run on test data.
if args.cuda:
    # first free up some GPU memory
    del train_data_tensor, train_target_tensor, val_data_tensor, val_target_tensor
    all_data_tensor = all_data_tensor.cuda()
    all_target_tensor = all_target_tensor.cuda()
test_loss, test_accuracy, high_accuracy, high_percentage = test()
print('=' * 89)
print('| End of training | test loss {:5.2f} | test accuracy {:.2f} | high confidence accuracy {:.2f} '
      '| high confidence percentage {:.2f} |'.format(
       test_loss, test_accuracy, high_accuracy, high_percentage))
print('=' * 89)

# acc = plt.figure()
# plt.plot(test_accuracy, 'g')
# acc.set_size_inches(32,18)
# plt.savefig('./{0} Plots/{1}_Batchsize{2}_Dropout{3}_LR{4}_Data{5}_Hidden{7}_Accuracy{6:.2f}.png'.format(
#             name, args.load_epochs+args.epochs, args.batch_size, args.dropout, args.lr, 'data', test_accuracy[-1], args.nhid),
#             dpi=250)
# plt.savefig('./{0} Plots/svg/{1}_Batchsize{2}_Dropout{3}_LR{4}_Data{5}_Hidden{7}_Accuracy{6:.2f}.svg'.format(
#             name, args.load_epochs+args.epochs, args.batch_size, args.dropout, args.lr, 'data', test_accuracy[-1], args.nhid),
#             dpi=250)

# los = plt.figure()
# plt.plot(training_losses,'-b', label='Training Losses', linestyle=':')
# plt.plot(validation_losses,'-g', label='Validation Losses')
# los.set_size_inches(32,18)
# plt.savefig('./{0} Plots/{1}_Batchsize{2}_Dropout{3}_LR{4}_Data{5}_Hidden{6}.png'.format(
#             name, args.load_epochs+args.epochs, args.batch_size, args.dropout, args.lr, data, hidden_dim),
#             dpi=250)
# plt.savefig('./{0} Plots/svg/{1}_Batchsize{2}_Dropout{3}_LR{4}_Data{5}_Hidden{6}.svg'.format(
#             name, args.load_epochs+args.epochs, args.batch_size, args.dropout, args.lr, data, hidden_dim),
#             dpi=250)

# plt.show() 
