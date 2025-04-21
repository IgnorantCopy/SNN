#---------------------------------------------------
# Imports
#---------------------------------------------------
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np
import datetime
import sys
import os
import argparse
from ann2snn.net import ConvNet

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def find_threshold(batch_size=512, time_steps=2500):
    
    loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    try:
        obj = model.module
    except AttributeError:
        obj = model
    
    obj.network_update(timesteps=time_steps, leak=1.0)
    

    pos=0
    thresholds=[]
    
    def find(layer, pos):
        max_act=0
        
        f.write('\n Finding threshold for layer {}'.format(layer))
        for batch_idx, (data, target) in enumerate(loader):
            
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=layer)
                if output>max_act:
                    max_act = output.item()

                #f.write('\nBatch:{} Current:{:.4f} Max:{:.4f}'.format(batch_idx+1,output.item(),max_act))
                if batch_idx==0:
                    thresholds.append(max_act)
                    pos = pos+1
                    f.write(' {}'.format(thresholds))
                    obj.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                    break
        return pos

    for l in obj.features.named_children():
        if isinstance(l[1], nn.Conv2d):
            pos = find(int(l[0]), pos)

    for c in obj.classifier.named_children():
        if isinstance(c[1], nn.Linear):
            if (int(l[0])+int(c[0])+1) == (len(obj.features) + len(obj.classifier) -1):
                pass
            else:
                pos = find(int(l[0])+int(c[0])+1, pos)

    f.write('\n ANN thresholds: {}'.format(thresholds))
    return thresholds

def train(epoch):

    global learning_rate

    model.module.network_update(timesteps=time_steps, leak=leak)
    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    if epoch in lr_interval:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / lr_reduce
            learning_rate = param_group['lr']
    
    #f.write('Epoch: {} Learning Rate: {:.2e}'.format(epoch,learning_rate_use))
    
    #total_loss = 0.0
    #total_correct = 0
    model.train()
    #model.module.network_init(update_interval)

    for batch_idx, (data, target) in enumerate(train_loader):
               
        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data) 
        #pdb.set_trace()
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()        
        pred = output.max(1,keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()

        losses.update(loss.item(),data.size(0))
        top1.update(correct.item()/data.size(0), data.size(0))
                
        if (batch_idx+1) % train_acc_batches == 0:
            temp1 = []
            for value in model.module.threshold.values():
                temp1 = temp1+[round(value.item(),2)]
            f.write('\nEpoch: {}, batch: {}, train_loss: {:.4f}, train_acc: {:.4f}, threshold: {}, leak: {}, time_steps: {}'
                    .format(epoch,
                        batch_idx+1,
                        losses.avg,
                        top1.avg,
                        temp1,
                        model.module.leak.item(),
                        model.module.timesteps
                        )
                    )
    f.write('\nEpoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}, time: {}'
                    .format(epoch,
                        learning_rate,
                        losses.avg,
                        top1.avg,
                        datetime.timedelta(seconds=(datetime.datetime.now() - train_start_time).seconds),
                        )
                    )


def test(epoch):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    with torch.no_grad():
        model.eval()
        global max_accuracy
        
        for batch_idx, (data, target) in enumerate(test_loader):
                        
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            
            output  = model(data) 
            loss    = F.cross_entropy(output,target)
            pred    = output.max(1,keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()

            losses.update(loss.item(),data.size(0))
            top1.update(correct.item()/data.size(0), data.size(0))
            
            if test_acc_every_batch:
                
                f.write('\nAccuracy: {}/{}({:.4f})'
                    .format(
                    correct.item(),
                    data.size(0),
                    top1.avg
                    )
                )
        
        temp1 = []
        for value in model.module.threshold.values():
            temp1 = temp1+[value.item()]    
        
        if epoch>5 and top1.avg<0.15:
            f.write('\n Quitting as the training is not progressing')
            exit(0)

        if top1.avg>max_accuracy:
            max_accuracy = top1.avg
             
            state = {
                    'accuracy'              : max_accuracy,
                    'epoch'                 : epoch,
                    'state_dict'            : model.state_dict(),
                    'optimizer'             : optimizer.state_dict(),
                    'thresholds'            : temp1,
                    'times_steps'            : time_steps,
                    'leak'                  : leak,
                    'activation'            : activation
                }
            try:
                os.mkdir('./trained_models/snn/')
            except OSError:
                pass 
            filename = './trained_models/snn/'+identifier+'.pth'
            torch.save(state,filename)    
        
            #if is_best:
            #    shutil.copyfile(filename, 'best_'+filename)

        f.write(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f} time: {}'
            .format(
            losses.avg, 
            top1.avg,
            max_accuracy,
            datetime.timedelta(seconds=(datetime.datetime.now() - test_start_time).seconds)
            )
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='MNIST',            type=str,       help='dataset name', choices=['MNIST',])
    parser.add_argument('--dataset_root',           default='E:/DataSets/',     type=str,       help='dataset root directory')
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('-lr','--learning_rate',    default=1e-4,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained ANN model')
    parser.add_argument('--pretrained_snn',         default='',                 type=str,       help='pretrained SNN for inference')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('--epochs',                 default=300,                type=int,       help='number of training epochs')
    parser.add_argument('--lr_interval',            default='0.60 0.80 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
    parser.add_argument('--time_steps',             default=100,                type=int,       help='simulation time_steps')
    parser.add_argument('--leak',                   default=1.0,                type=float,     help='membrane leak')
    parser.add_argument('--scaling_factor',         default=0.7,                type=float,     help='scaling factor for thresholds at reduced time_steps')
    parser.add_argument('--default_threshold',      default=1.0,                type=float,     help='initial threshold to train SNN from scratch')
    parser.add_argument('--activation',             default='Linear',           type=str,       help='SNN activation function', choices=['Linear', 'STDB'])
    parser.add_argument('--alpha',                  default=0.3,                type=float,     help='parameter alpha for STDB')
    parser.add_argument('--beta',                   default=0.01,               type=float,     help='parameter beta for STDB')
    parser.add_argument('--optimizer',              default='Adam',             type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay',           default=5e-4,               type=float,     help='weight decay parameter for the optimizer')    
    parser.add_argument('--momentum',               default=0.95,                type=float,     help='momentum parameter for the SGD optimizer')    
    parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--dropout',                default=0.3,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--test_acc_every_batch',   action='store_true',                        help='print acc of every batch during inference')
    parser.add_argument('--train_acc_batches',      default=200,                type=int,       help='print training progress after this many batches')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    
    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
           
    dataset             = args.dataset
    dataset_root        = args.dataset_root
    batch_size          = args.batch_size
    learning_rate       = args.learning_rate
    pretrained_ann      = args.pretrained_ann
    pretrained_snn      = args.pretrained_snn
    epochs              = args.epochs
    lr_reduce           = args.lr_reduce
    time_steps           = args.timesteps
    leak                = args.leak
    scaling_factor      = args.scaling_factor
    default_threshold   = args.default_threshold
    activation          = args.activation
    alpha               = args.alpha
    beta                = args.beta  
    optimizer           = args.optimizer
    weight_decay        = args.weight_decay
    momentum            = args.momentum
    amsgrad             = args.amsgrad
    dropout             = args.dropout
    kernel_size         = args.kernel_size
    test_acc_every_batch= args.test_acc_every_batch
    train_acc_batches   = args.train_acc_batches

    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs))

    log_file = './logs/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass 

    #identifier = 'snn_'+architecture.lower()+'_'+dataset.lower()+'_'+str(time_steps)+'_'+str(datetime.datetime.now())
    identifier = pretrained_ann.split('/')[-1].split('.')[0]
    log_file = os.path.join(log_file, 'log_' + identifier)
    _, dataset_name, batch_size, optimizer, learning_rate = identifier.split('_')
    
    if args.log:
        f = open(log_file, 'w', buffering=1)
    else:
        f = sys.stdout

    if not pretrained_ann:
        ann_file = './trained_models/ann/ann_'+dataset.lower()+'.pth'
        if os.path.exists(ann_file):
            val = input('\n Do you want to use the pretrained ANN {}? Y or N: '.format(ann_file))
            if val.lower()=='y' or val.lower()=='yes':
                pretrained_ann = ann_file

    f.write('\n Run on time: {}'.format(datetime.datetime.now()))

    f.write('\n\n Arguments: ')
    for arg in vars(args):
        if arg == 'lr_interval':
            f.write('\n\t {:20} : {}'.format(arg, lr_interval))
        elif arg == 'pretrained_ann':
            f.write('\n\t {:20} : {}'.format(arg, pretrained_ann))
        else:
            f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))
    f.flush()
    # Training settings
    
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    normalize       = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    
    if dataset == 'MNIST':
        trainset   = datasets.MNIST(root=dataset_root, train=True, download=True, transform=transforms.ToTensor())
        testset    = datasets.MNIST(root=dataset_root, train=False, download=True, transform=transforms.ToTensor())
        image_size = 28
        labels = 10
    else:
        raise NotImplementedError('Only MNIST is supported.')

    train_loader    = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader     = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = ConvNet(labels, image_size, batch_size, dropout=dropout, num_channels=1)

    # if freeze_conv:
    #     for param in model.features.parameters():
    #         param.requires_grad = False
    
    #Please comment this line if you find key mismatch error and uncomment the DataParallel after the if block
    model = nn.DataParallel(model)
    
    if pretrained_ann:
        state = torch.load(pretrained_ann, map_location='cpu')
        cur_dict = model.state_dict()     
        for key in state['state_dict'].keys():
            if key in cur_dict:
                if (state['state_dict'][key].shape == cur_dict[key].shape):
                    cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
                    f.write('\n Success: Loaded {} from {}'.format(key, pretrained_ann))
                else:
                    f.write('\n Error: Size mismatch, size of loaded model {}, size of current model {}'.format(
                        state['state_dict'][key].shape, model.state_dict()[key].shape))
            else:
                f.write('\n Error: Loaded weight {} not present in current model'.format(key))
        model.load_state_dict(cur_dict)
        f.write('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))

        #If thresholds present in loaded ANN file
        if 'thresholds' in state.keys():
            thresholds = state['thresholds']
            f.write('\n Info: Thresholds loaded from trained ANN: {}'.format(thresholds))
            try :
                model.module.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
            except AttributeError:
                model.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
        else:
            thresholds = find_threshold(batch_size=512, time_steps=1000)
            try:
                model.module.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
            except AttributeError:
                model.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
            
            #Save the thresholds in the ANN file
            temp = {}
            for key,value in state.items():
                temp[key] = value
            temp['thresholds'] = thresholds
            torch.save(temp, pretrained_ann)
    
    if pretrained_snn:
                
        state = torch.load(pretrained_snn, map_location='cpu')
        cur_dict = model.state_dict()     
        for key in state['state_dict'].keys():
            
            if key in cur_dict:
                if (state['state_dict'][key].shape == cur_dict[key].shape):
                    cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
                    f.write('\n Loaded {} from {}'.format(key, pretrained_snn))
                else:
                    f.write('\n Size mismatch {}, size of loaded model {}, size of current model {}'.format(
                        key, state['state_dict'][key].shape, model.state_dict()[key].shape))
            else:
                f.write('\n Loaded weight {} not present in current model'.format(key))
        model.load_state_dict(cur_dict)

        if 'thresholds' in state.keys():
            try:
                if state['leak_mem']:
                    state['leak'] = state['leak_mem']
            except:
                pass
            if state['time_steps']!=time_steps or state['leak']!=leak:
                f.write('\n Time_steps/Leak mismatch between loaded SNN and current simulation time_steps/leak, '
                        'current time_steps/leak {}/{}, loaded time_steps/leak {}/{}'.format(
                    time_steps, leak, state['time_steps'], state['leak']))
            thresholds = state['thresholds']
            model.module.threshold_update(scaling_factor = state['scaling_threshold'], thresholds=thresholds[:])
        else:
            f.write('\n Loaded SNN model does not have thresholds')

    f.write('\n {}'.format(model))
    f.flush()

    if torch.cuda.is_available() and args.gpu:
        model.cuda()

    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    
    f.write('\n {}'.format(optimizer))
    max_accuracy = 0
    
    for epoch in range(1, epochs):
        train_start_time = datetime.datetime.now()
        if not args.test_only:
            train(epoch)
        test_start_time = datetime.datetime.now()
        test(epoch)

    f.write('\n Highest accuracy: {:.4f}'.format(max_accuracy))
    f.close()
