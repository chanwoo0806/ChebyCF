import argparse, yaml
import random, numpy, torch
import logging, os, time

def get_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--config', type=str, default='default')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--batch_size', type=int)     
    # Dataset, Metric
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--metrics', type=str)
    parser.add_argument('--top_ks', type=str)
    # Model
    parser.add_argument('--model', type=str, help='ChebyCF, GFCF')
    parser.add_argument('--K', type=int, help='Order of Chebyshev Filter')
    parser.add_argument('--phi', type=float, help='Flatness of Plateau Transfer Function')
    parser.add_argument('--eta', type=int, help='Threshold Frequency of Ideal Pass Filter')
    parser.add_argument('--alpha', type=float, help='Weight of Ideal Pass Filter')
    parser.add_argument('--beta', type=float, help='Power of Degree-based Normalization')
    args = parser.parse_args()
    
    # Use config file to fill in missing arguments
    with open(f'./config/{args.config}.yml', mode='r', encoding='utf-8') as f:
        config = yaml.safe_load(f.read())
    for arg, val in vars(args).items():
        if (val is None) and (arg in config):
            setattr(args, arg, config[arg])
    
    # Convert comma-separated string to list
    def str_to_list(string, elem_type):
        return [elem_type(x) for x in string.split(",")]
    def is_str(x):
        return isinstance(x, str)
    args.metrics = str_to_list(args.metrics, str) if is_str(args.metrics) else args.metrics
    args.top_ks = str_to_list(args.top_ks, int) if is_str(args.top_ks) else args.top_ks
    return args

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_logger(comment):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter(fmt='%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # Log to console
    strm_handler = logging.StreamHandler()
    strm_handler.setFormatter(formatter)
    logger.addHandler(strm_handler)
    # Log to file
    if not os.path.exists('./log'): os.makedirs('./log')
    comment = '-' + comment if comment else ''
    log_file = f'./log/{time.strftime("%y%m%d-%H%M%S")}{comment}.log'
    file_handler = logging.FileHandler(log_file, 'a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def log_args(args):
    logging.info('[SETTING]')
    for arg, val in vars(args).items():
        if val is not None:
            logging.info(f'{arg}: {val}')
    logging.info('')