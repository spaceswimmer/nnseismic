from .base_options import BaseOptions3d

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class TrainOptions3d(BaseOptions3d):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions3d.initialize(self, parser)
        # training parameters
        parser.add_argument('--nepochs', type=int, default=100, help='# of epochs for training')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay rate for adam')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
        parser.add_argument('--lr_factor', default=0.1, type=float, help='learning rate decay factor')
        parser.add_argument('--lr_patience', default=4, type=int, help='learning rate patience')
        parser.add_argument('--pretrained_model',type=str,default=None,help='load pretrained model')
        parser.add_argument('--data_augmentation',type=str2bool,default=True,help='use data augmentation')
        parser.add_argument('--valid',type=str2bool,default=False,help='valid phase')
        parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between saving model weights")
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument("--valid_interval", type=int, default=100, help="interval between validation")
        parser.add_argument("--history_interval", type=int, default=50, help="interval between output loss history")
        parser.add_argument('--dataroot_val',type=str, default=None, help='''path to data for validation''')
        parser.add_argument('--loss_type',type=str, default='MSE', help='''loss function''')
        parser.add_argument('--dataset_size_val', type=int, default=float("inf"), help='''The size of the dataset for validation''')    
        self.isTrain = True
        return parser
