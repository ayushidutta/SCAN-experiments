from data_loader import load_data_v2, get_data_iters, EOS_TOKEN
from solver_v2 import load_model, train, test
import os
import argparse
import torch

parser = argparse.ArgumentParser(description='SCAN reproduction')
parser.add_argument('--data_dir', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--train_path', default='', type=str, metavar='PATH',
                    help='path to train dataset (e.g. experiment1/train')
parser.add_argument('--test_path', default='', type=str, metavar='PATH',
                    help='path to test dataset')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--hidden_dim', default=200, type=int,
                    metavar='N', help='LSTM Hidden Dimension (default: 200)')
parser.add_argument('--dropout', default=0, type=float, metavar='D',
                    help='dropout')
parser.add_argument('--bidirection', default=0, type=int, metavar='BOOL',
                    help='Encoder is bidirectional')
parser.add_argument('--num_layers', default=2, type=int,
                    metavar='N', help='Num of layers in RNN')
parser.add_argument('--rnn_type', default='lstm', type=str, metavar='LSTM',
                    help='RNN Tpe is lstm/gru (default: LSTM)')
parser.add_argument('--model_dir', default='', type=str, metavar='DIR',
                    help='path to model directory (default: none)')
parser.add_argument('--add_pos', default=0, type=int, metavar='BOOL',
                    help='Will add POS tags (default: False)')
parser.add_argument('--add_dl', default=0, type=int, metavar='BOOL',
                    help='Will add DL tags (default: False)')
parser.add_argument('--cl', default=0, type=int, metavar='BOOL',
                    help='Curriculum Learning is true (default: False)')
parser.add_argument('--eval', default=0, type=int, metavar='BOOL',
                    help='Will Evaluate (default: False)')


def main():
    # Define Arguments
    args = parser.parse_args()
    path_train = os.path.join(args.data_dir, args.train_path)
    path_test = os.path.join(args.data_dir, args.test_path)
    model_dir = args.model_dir
    model_path = os.path.join(args.model_dir, 'model_100000.pt')
    eval = args.eval
    batch_size = args.batch_size
    add_pos = True if args.add_pos==1 else False
    add_dl = True if args.add_dl==1 else False
    cl = True if args.cl==1 else False
    state = {
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'bidirection': True if args.bidirection==1 else False,
        'num_layers': args.num_layers,
        'rnn_type': args.rnn_type,
        'add_pos': add_pos,
        'add_dl': add_dl
    }
    print(f"Run Config State, Eval: {state}, {eval}")
    # Train and Test
    train_data, test_data, data_fields = load_data_v2(path_train, path_test, model_dir, add_pos=add_pos, add_dl=add_dl)
    #data_iters = get_data_iters(train_data, test_data, batch_size=batch_size, cl=cl)
    #model, optimizer, criterion = load_model(data_fields, state)
    #if eval == 0:
        #if cl:
            #print('Training by Curriculum Learning!')
            #pass
        #else:
            #print('Training !')
            #model = train(model, data_iters[0], optimizer, criterion, model_dir=model_dir)
    #else:
        #print('Evaluating !')
        #model.load_state_dict(torch.load(model_path))
    #test(model, data_iters[-1], eos_index=data_fields["trg"].vocab.stoi[EOS_TOKEN])

if __name__ == "__main__":
	main()