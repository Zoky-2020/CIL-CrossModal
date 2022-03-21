from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from dataset import TestData
from dataset.data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
from corruption import RandomCorruption

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')


parser.add_argument('--arch', default='resnet50', type=str, help='network baseline: resnet50')
parser.add_argument('--workers', default=4, type=int, metavar='N')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size') 

parser.add_argument('--method', default='awg', type=str,  metavar='m', help='method type: base or awg')
parser.add_argument('--margin', default=0.3, type=float,  metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')


parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--model_path', default='.ckpt/', type=str, help='model save path')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')

parser.add_argument('--corrupt',action= 'store_true', help= 'corruption')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


dataset = args.dataset
if dataset == 'sysu':
    data_path = './SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]
elif dataset =='regdb':
    data_path = './RegDB/'
    n_class = 206
    test_mode = [2, 1]
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pool_dim = 2048
print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch=args.arch)
else:
    net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch)
net.to(device)    
cudnn.benchmark = True

checkpoint_path = args.model_path

if args.method =='id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

transform_test_c = transforms.Compose([
    transforms.ToPILImage(),
    RandomCorruption(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()


def extract_gall_feat(gall_loader):
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall  , pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, test_mode[0])
            gall_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_pool, gall_feat_fc
    
def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, test_mode[1])
            query_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_pool, query_feat_fc



if dataset == 'sysu':

    model_path = checkpoint_path
    if os.path.isfile(model_path):
        print('==> loading checkpoint ...')
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint at (epoch {})'.format(checkpoint['epoch']))
    else:
        print('==> no checkpoint found')

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))


    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_feat_pool, query_feat_fc = extract_query_feat(query_loader)


    trial_gallset_c = TestData(gall_img, gall_label, transform=transform_test_c, img_size=(args.img_w, args.img_h))        
    trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    
    trial_gall_loader_c = data.DataLoader(trial_gallset_c, batch_size=args.test_batch, shuffle=False, num_workers=4)
    trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    
    gall_feat_pool_c, gall_feat_fc_c = extract_gall_feat(trial_gall_loader_c)
    gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader)

    # pool5 feature
    distmat_pool_c = np.matmul(query_feat_pool, np.transpose(gall_feat_pool_c))
    cmc_pool_c, mAP_pool_c, mINP_pool_c = eval_sysu(-distmat_pool_c, query_label, gall_label, query_cam, gall_cam)

    distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
    cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

    print('Clean:')
    print('mINP: {:.4f}, mAP: {:.4f}, R1: {:.4f}, R5: {:.4f}, R10: {:.4f}'.format(mINP_pool, mAP_pool, cmc_pool[0], cmc_pool[4], cmc_pool[9]))
    print('Corruption:')
    print('mINP: {:.4f}, mAP: {:.4f}, R1: {:.4f}, R5: {:.4f}, R10: {:.4f}'.format(mINP_pool_c, mAP_pool_c, cmc_pool_c[0], cmc_pool_c[4], cmc_pool_c[9]))          


elif dataset == 'regdb':

    model_path = checkpoint_path
    if os.path.isfile(model_path):
        print('==> loading checkpoint ...')
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])

    test_trial =args.trial 
        
    # testing set
    # the para. 'trial' for eval. should keep the same as training
    query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')

    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    nquery = len(query_label)
    ngall = len(gall_label)

    queryset_c = TestData(query_img, query_label, transform=transform_test_c, img_size=(args.img_w, args.img_h))        
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

    query_loader_c = data.DataLoader(queryset_c, batch_size=args.test_batch, shuffle=False, num_workers=4)        
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)


    query_feat_pool_c, query_feat_fc_c = extract_query_feat(query_loader_c)
    query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
    gall_feat_pool,  gall_feat_fc = extract_gall_feat(gall_loader)

    if args.tvsearch:
        # pool5 feature
        distmat_pool_c = np.matmul(gall_feat_pool, np.transpose(query_feat_pool_c))
        cmc_pool_c, mAP_pool_c, mINP_pool_c = eval_regdb(-distmat_pool_c, gall_label, query_label)

        distmat_pool = np.matmul(gall_feat_pool, np.transpose(query_feat_pool))
        cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, gall_label, query_label)

    else:
        # pool5 feature
        distmat_pool_c = np.matmul(query_feat_pool_c, np.transpose(gall_feat_pool))
        cmc_pool_c, mAP_pool_c, mINP_pool_c = eval_regdb(-distmat_pool_c, query_label, gall_label)

        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, query_label, gall_label)

    print('Clean:')
    print('mINP: {:.4f}, mAP: {:.4f}, R1: {:.4f}, R5: {:.4f}, R10: {:.4f}'.format(mINP_pool, mAP_pool, cmc_pool[0], cmc_pool[4], cmc_pool[9]))
    print('Corruption:')
    print('mINP: {:.4f}, mAP: {:.4f}, R1: {:.4f}, R5: {:.4f}, R10: {:.4f}'.format(mINP_pool_c, mAP_pool_c, cmc_pool_c[0], cmc_pool_c[4], cmc_pool_c[9]))          
