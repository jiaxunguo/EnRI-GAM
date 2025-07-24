import os
import sys
import torch
import numpy as np
import datetime
import logging
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', type=bool, default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='rinet', help='model name [default: pabftnetv2_cls]')
    parser.add_argument('--num_category', default=15, type=int, help='training on scanobj')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--data_type', type=str, default='OBJ_BG', help='data type')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_uniform_sample', type=bool, default=True, help='use uniform sampiling')
    
    parser.add_argument('--pretrain_weight', type=str, default=None)
    
    parser.add_argument('--trainset', type=str, default='z', choices=['z', 'so3'], help='Rotation for trainset')
    parser.add_argument('--testset', type=str, default='z', choices=['z', 'so3'], help='Rotation for testset')
    
    parser.add_argument('--dp_rate', type=float, default=0.5, help='for step_LR')
    parser.add_argument('--k', type=int, default=20, help='dirname')
    parser.add_argument('--output_channels', type=int, default=15, help='num_category')
    
    
    return parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    
def test(model, loader,args, num_class=40):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    # for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
    #     if not args.use_cpu:
    #         points, target = points.cuda(), target.cuda()  
    #     pred, _ = classifier(points)
    #     if len(pred.shape) == 3:
    #         pred = pred.mean(dim=1)
    #     pred_choice = pred.data.max(1)[1]
        
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            data = data.cuda()
            target = data.y.cuda()
        
        net_outputs = classifier(data)
        pred_choice = net_outputs["x"].max(dim=1)[1]
        
        for i in range(len(target)):
            class_acc[target[i],0]+=1
            if target[i]==pred_choice[i]:
                class_acc[pred_choice[i],1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(target.size()[0]))
    class_acc[:, 2] = class_acc[:, 1] / class_acc[:, 0]
    in_average=sum(class_acc[:, 1])/sum(class_acc[:, 0])
    cla_average=sum(class_acc[:, 2])/len(class_acc[:, 2])
    return in_average, cla_average

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    from models.rinet import RINet
    from data_utils.ScanObjectNNLoaderGeo import ScanObjectNNGeo
    from torch_geometric.data import DataLoader
    
    torch.autograd.set_detect_anomaly(True)
    
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification_scanobj')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    '''SUMMARY WRITER'''
    writer = SummaryWriter(log_dir)
    log_hyperparams(writer, args)
    
    '''DATA LOADING'''
    log_string('Load dataset ...')
    
    if args.data_type == 'OBJ_NOBG':
        data_path = os.path.join(ROOT_DIR, 'data/scanobjectnn/main_split_nobg/')
    elif args.data_type == 'hardest' or 'OBJ_BG': 
        data_path = os.path.join(ROOT_DIR, 'data/scanobjectnn/main_split/')
    else:
        raise NotImplementedError()
    
    
    
    train_dataset = ScanObjectNNGeo(root=data_path, args=args, split='train', transform=args.trainset) 
    test_dataset = ScanObjectNNGeo(root=data_path, args=args, split='test', transform=args.testset)
    
    trainDataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
    
    '''MODEL LOADING'''
    num_class = args.num_category
    # model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('./data_utils/ScanObjectNNLoaderGeo.py', str(exp_dir))
    shutil.copy('./train_classification_scanobj.py', str(exp_dir))

    # classifier = model.get_model(num_class, 1)
    # criterion = model.get_loss()
    # classifier.apply(inplace_relu)
    
    if args.model == 'rinet':
        classifier = RINet(args)
    else:
        raise NotImplementedError()
    
    criterion = classifier.get_loss
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
    
    if args.pretrain_weight:
        print("Load initial weight from %s\n" % args.pretrain_weight)
        checkpoint = torch.load(args.pretrain_weight)
        classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
        for name, param in classifier.named_parameters():
            if name in checkpoint['model_state_dict']:
                print(name)
                
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
        eta_min = 1e-5
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate*100, momentum=0.9)      
        eta_min = 1e-3

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=eta_min)
    
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    log_string('Trainable Parameters: %f' % (count_parameters(classifier)))

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()
        
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            if not args.use_cpu:
                data = data.cuda()
                target = data.y.cuda()

            net_outputs = classifier(data)
            loss = criterion(net_outputs, target, writer, global_step + 1, global_epoch + 1, label_smoothing=0.2)
            
            pred_choice = net_outputs["x"].max(dim=1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(target.size()[0]))
            
            loss.backward()
            
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        # if epoch==300: # Warm up, a popular transformer training scheme
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = args.learning_rate
        
        writer.add_scalar('train/epoch-loss', loss, global_epoch)
        writer.add_scalar('train/train_instance_acc', train_instance_acc, global_epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_epoch)
        writer.flush()
        
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        log_string('Train loss: %f' % loss)
        log_string('lr: %f' % optimizer.param_groups[0]['lr'])
        log_string('Bingham paras: %s' % np.array2string(net_outputs["B_paras"].detach().cpu().numpy(), precision=3, separator=' '))
        scheduler.step()

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, args, num_class=num_class)
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
                
            writer.add_scalar('test/instance_acc', instance_acc, global_epoch)
            writer.add_scalar('test/class_acc', class_acc, global_epoch)
            writer.flush()
            
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
            
        classifier.update_binghamer()
        
    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
