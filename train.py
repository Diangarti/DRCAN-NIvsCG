import argparse
import numpy as np
import chainer
from chainer import training, datasets, iterators, optimizers, serializers
from chainer import reporter
from network_conv import RAM
from chainer.training import extensions
from weightdecay import lr_drop
from config_dram import Config
from chainer.backends import cuda
import input_data

config  = Config()
#data_dir = 'E:/Diangarti/CSI_Dataset/Dataset_1'
data_dir = 'E:/Diangarti/CGI_PG_LSTM/Dataset_Patch_240'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAM in Chainer:MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=4000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result_240_no_conv_0_amsgrad',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=128,
                        help='Dimension of locator, glimpse hidden state')
    parser.add_argument('--hidden','-hi', type=int, default=256,
                        help='Dimension of lstm hidden state')
    parser.add_argument('--g_size', '-g_size', type=int, default=8,
                        help='Dimension of output')
    parser.add_argument('--len_seq', '-l', type=int, default=6,
                        help='Length of action sequence')
    parser.add_argument('--depth', '-d', type=int, default=1,
                        help='no of depths/glimpses to be taken at once')
    parser.add_argument('--scale', '-s', type=float, default=2,
                        help='subsequent scales of cropped image for sequential depths (int>1)')
    parser.add_argument('--sigma', '-si',type=float, default=0.03,
                        help='sigma of location sampling model')
    parser.add_argument('--evalm', '-evalm', type=str, default=None,
                        help='Evaluation mode: path to saved model file')
    parser.add_argument('--evalo', '-eval0', type=str, default=None,
                        help='Evaluation mode: path to saved optimizer file')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# n_units: {}'.format(args.unit))
    print('# n_hidden: {}'.format(args.hidden))
    print('# Length of action sequence: {}'.format(args.len_seq))
    print('# sigma: {}'.format(args.sigma))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    data = input_data.read_data_sets(data_dir,one_hot=False,config = config)
    train_data= data.train.images
    test_data= data.test.images
    train_targets = data.train.labels
    test_targets = data.test.labels
    train= datasets.tuple_dataset.TupleDataset(train_data, train_targets)
    test= datasets.tuple_dataset.TupleDataset(test_data, test_targets)
    print(train)
    #train, test = chainer.datasets.get_mnist()
    #print(train)
    train_data, train_targets = np.array(train).transpose()
    test_data, test_targets = np.array(test).transpose()
    print(np.array(train_data.shape))
    train_data = np.array(list(train_data)).reshape(train_data.shape[0], 3, 240, 240)
    test_data = np.array(list(test_data)).reshape(test_data.shape[0], 3, 240, 240)
    train_targets = np.array(train_targets).astype(np.int32)
    test_targets = np.array(test_targets).astype(np.int32)
#    if args.evalm is not None:
#        chainer.global_config.train = False

#    model = RAM( args.hidden, args.unit, args.sigma,
#                 args.g_size, args.len_seq, args.depth, args.scale, n_in = 32,using_conv = True)
    model = RAM( args.hidden, args.unit, args.sigma,
                 args.g_size, args.len_seq, args.depth, args.scale, using_conv = True)
    #model.to_gpu()
    #optimizer = optimizers.NesterovAG()
    optimizer = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08, eta=1.0, weight_decay_rate=0, amsgrad=True)
    if args.evalm is not None:
        serializers.load_npz(args.evalm, model)
        print('model loaded')
#    if args.evalo is not None:
#        serializers.load_npz(args.evalo, trainer)
#        print('optimizer loaded')

    if args.gpu>=0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer.setup(model)
    print("hhh")
    train_dataset = datasets.tuple_dataset.TupleDataset(train_data, train_targets)
    train_iter = iterators.SerialIterator(train_dataset, args.batchsize)
    test_dataset = datasets.tuple_dataset.TupleDataset(test_data, test_targets)
    test_iter = iterators.SerialIterator(test_dataset, 128, False, False)
    stop_trigger = (args.epoch, 'epoch')
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)
    if args.evalo is not None:
        serializers.load_npz(args.evalo, trainer)
        print('trainer loaded')
    #trainer.extend(lr_drop)
    trainer.extend(extensions.LogReport(log_name='log'))
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}.npz'), trigger=(50,'epoch'))
    #trainer.extend(extensions.snapshot_object(model, filename='model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))

    trainer.extend(extensions.snapshot_object(model, 'my.model{.updater.epoch}'), trigger=(50,'epoch'))
    trainer.extend(extensions.snapshot_object(optimizer, 'my.optimizer{.updater.epoch}'), trigger=(50, 'epoch'))
    trainer.extend(extensions.snapshot_object(trainer, 'my.trainer{.updater.epoch}'), trigger=(50, 'epoch'))
    trainer.extend(extensions.PlotReport(['main/cross_entropy_loss', 'validation/main/cross_entropy_loss'], x_key='epoch',trigger=(1, 'epoch'), file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch',trigger=(1, 'epoch'), file_name='accuracy.png'))

    trainer.extend(extensions.PlotReport(['main/accuracy'], 'epoch', trigger=(1, 'epoch'), file_name='train_accuracy.png',
                          marker="."))
    trainer.extend(extensions.PlotReport(['main/cross_entropy_loss'], 'epoch', trigger=(1, 'epoch'), file_name='train_cross_entropy.png',
                          marker="."))
    #trainer.extend(extensions.ProgressBar((args.epoch,'epoch'),update_interval=50))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()
    serializers.save_npz('my.model', model)
    serializers.save_npz('my.optimizer', optimizer)
    serializers.save_npz('my.trainer', trainer)