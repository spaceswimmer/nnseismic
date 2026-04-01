import argparse
import torch
from datetime import datetime

class BaseOptions3d():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and
    saving the options.
    """
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot',
                            required=True,
                            help='''path to data (should have subfolders seis,
                            rgt)
                            ''')
        parser.add_argument('--session_name', 
                            type=str, action="store", 
                            default=datetime.now().strftime('%b%d_%H%M%S'),
                            help="name of the session to be ised in saving the model")
        parser.add_argument('--gpu_ids',
                    type=str,
                            default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--shape',
                            type=int,
                            nargs=3,
                            help='the shape of the input and target')
        parser.add_argument('--n_channels',
                            type=int,
                            default=1,
                            help='''# of input image channels: 3 for RGB and
                            1 for grayscale
                            ''')
        parser.add_argument('--batch_size',
                            type=int,
                            default=1,
                            help='input batch size')
        parser.add_argument('--dataset_size',
                            type=int,
                            default=float("inf"),
                            help='''The size of the dataset''')
        parser.add_argument('--num_workers',
                            type=int,
                            default=0,
                            help='''# number of CPU used to load data
                            ''')  
        parser.add_argument('--sessions_path',
                            type=str,
                            default='sessions',
                            help='sessions are saved here')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=\
                     argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = '=====================================================================================================\n'
        if self.isTrain:
            message += '\t\t\t\t\tTRAIN OPTIONS\n'
        else: 
            message += '\t\t\t\t\tTEST OPTIONS\n'
        message += '=====================================================================================================\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '=====================================================================================================\n'
        print(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix,
        and set up GPU device.
        """
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)

        self.opt = opt
        return self.opt
