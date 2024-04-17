import torch
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)

class embedder:
    def __init__(self, args):
        self.embedder = args.embedder
        self.iter_max = args.iter_max
        self.dim = args.dim
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.gpu_num = args.gpu_num
        self.isInit = args.isInit
        self.reg_coef = args.reg_coef
        self.patience = args.patience
        self.num_aspects = args.num_aspects
        if self.gpu_num == -1:
            self.device = "cpu"
        else:
            self.device = torch.device("cuda:" + str(self.gpu_num) if torch.cuda.is_available() else "cpu")
        args.device = self.device
        self.tau_gumbel = args.tau_gumbel
