from .base_options import BaseOptions3d

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class TestOptions3d(BaseOptions3d):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """
    def initialize(self, parser):
        parser = BaseOptions3d.initialize(self, parser)
        # training parameters
        parser.add_argument('--trained_model',type=str,default=None,help='load trained model')
        parser.add_argument('--only_load_input',type=str2bool,default=True,help='only load input')
        self.isTrain = False
        return parser
