import argparse
from torch import optim
import torch.nn as nn
import torch
from lmmdModel import DSAN
import os
import sys
# from datasets.digit_provider import DataLoaderManager
from datasets.data_provider import DataLoaderManager

from tqdm import trange, tqdm
from torch.autograd import Variable
from evaluator import evaluate
import numpy as np
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from easydl.common import FileListDataset
from torchvision import transforms
from pretrainedmodel import SourceNet





DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


temperature = 2
LEARNING_RATE = 1e-3
IMAGE_SIZE = 28
N_EPOCH = 5
LOG_INTERVAL = 20

result = [] # Save the results


def select_score_high_tar (net, loader_tar, epoch):
    dataloader = loader_tar
    nc = 10
    alpha = 0
    with torch.no_grad():
        for X in dataloader:
            X = X.to(DEVICE)
            len = X.size
            net.eval()
            y_hat, _, _ = net(input_data=X, alpha=alpha)
            y_max = y_hat.argmax(dim=1)
            S = torch.ones(len)
            S[:] = threshold
            indies = np.argwhere(torch.gt(y_max, S)).reshape(-1)
            X_sel = X[indies][:, :]
            y_sel = nn.Softmax(dim=1)(y_hat).argmax(dim=1)[indies][:]


nc = 65
ncArray = list(range(65))
e_alpha = 0.5
def exploss(y_source_prob, y_source):
    loss_sum = 0
    for i in range(nc):
        index_i = y_source == i
        a = torch.exp(-1 * e_alpha * y_source_prob[index_i, i])
        b = 0
        for otheri in ncArray:

            if otheri == i:
                pass
            else:
                index_other = y_source == otheri
                ni = index_i.float().sum().item()
                nj = index_other.float().sum().item()
                if ni > 0 and nj > 0:
                    b += torch.sum(torch.exp(e_alpha * y_source_prob[index_other, i])) / (ni * nj)

        loss = torch.sum(a) * b
        loss_sum += loss

    return loss_sum



threshold = 0.7725
temperature = 2
lr = 0.01
res_auc = 0.0

def train(model_instance, data_loader_manager, soft_labels, optimizer, args=None ):
    totalNet = model_instance.cuda()
    train_stats = {}
    N_EPOCH = math.ceil(args.train_steps // args.batch_size)

    iter_num = 0

    total_progress_bar = tqdm(desc='Train iter', total=args.train_steps, ncols=100)
    
    for i_th_iter in trange(args.train_steps):


        epoch = math.ceil(i_th_iter / args.batch_size)
        if args.self_train is True and i_th_iter % args.yhat_update_freq == 0:
            print("第" + str(i_th_iter) + "次迭代，需要更新标签")
            data_loader_manager.update_self_training_labels(totalNet)
        source_loader, target_loader = data_loader_manager.get_train_source_target_loader()
        _, inputs_source, labels_source = next(iter(source_loader))
        _, inputs_target, labels_target = next(iter(target_loader))
        inputs_source, inputs_target = Variable(inputs_source).cuda(), Variable(inputs_target).cuda()
        labels_source = Variable(labels_source).cuda()
        labels_target = Variable(labels_target).cuda()
        
        
        if iter_num % args.eval_interval == 0 or iter_num == args.train_steps - 1:
            eval_result, prob_hat = evaluate.evaluate_from_dataloader(model_instance,
                                                                      data_loader_manager.get_test_target_loader())
            total_progress_bar.set_description(
                desc=f'Acc {eval_result["accuracy"]:.4f}, cls avg acc {eval_result["test_balanced_acc"]:.4f}')
            train_stats.update(eval_result)


        label_source_pred, loss_mmd, tar_y_hat = totalNet(inputs_source, inputs_target, labels_source)
     
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), labels_source)

        class_output = nn.Softmax(dim=1)(label_source_pred)
        loss_mauc = exploss(class_output, labels_source)

        tar_y_hat_softMax = nn.Softmax(dim=1)(tar_y_hat)

        y_max, _ = torch.max(tar_y_hat_softMax, dim=1)
        tar_y_output = tar_y_hat_softMax[y_max >= threshold, :]
        tar_sel_label = tar_y_hat_softMax.argmax(dim=1)[y_max >= threshold]
        tar_true_label_sel = labels_target[y_max >= threshold]
        acc_sel = (tar_sel_label == tar_true_label_sel).float().sum().item()
        n_sel = len(labels_target[y_max >= threshold])
        if n_sel > 0:
            sel_correct = acc_sel / n_sel
            print("挑选样本正确的准确率" + str(sel_correct))

        tar_y_hat_select = tar_y_hat[y_max >= threshold, :]

        err_t_auc = exploss(tar_y_output, tar_sel_label)
        soft_label_for_batch = ret_soft_label(tar_sel_label, soft_labels)
        soft_label_for_batch = soft_label_for_batch.to(DEVICE)
        output_cl_score = F.softmax(tar_y_hat_select / temperature, dim=1)

        loss_soft = 0.0
        if float(output_cl_score.size(0)) > 0:
            loss_soft = - (torch.sum(soft_label_for_batch * torch.log(output_cl_score))) / float(output_cl_score.size(0))   
        print("挑选的置信度的高的样本" + str(output_cl_score.size(0)))         



        lambd = 2 / (1 + math.exp(-10 * (epoch) / N_EPOCH)) - 1
        loss =  loss_cls +  0.3 * lambd * loss_mmd  + 0.01 * loss_mauc  +  0.25 * loss_soft
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_num += 1
        total_progress_bar.update(1)
        

def load_data_and_train(args):
    data_loader_manager = DataLoaderManager(args)
    return data_loader_manager


def run():
    parser = argparse.ArgumentParser()

    # training options
    parser.add_argument('--train_steps', default=10000, type=int, help='number of training steps')
    parser.add_argument('--eval_interval', default=50, type=int, help="eval interval for tensorboard")
    parser.add_argument('--train_loss', default='total_loss', type=str,
                        help="loss for training, e.g., total_loss, classifier_loss")

    # dataset
    parser.add_argument('--datasets_dir', type=str,
                        help='directory for all domain adaptation datasets')
    parser.add_argument('--dataset', default='Office-Home', type=str, help='Office-31, Office-Home, VisDA')
    parser.add_argument('--src_address', default=None, type=str, help='address of image list of source domain dataset')
    parser.add_argument('--tgt_address', default=None, type=str, help='address of image list of target domain dataset')
    parser.add_argument('--class_num', default=None, type=int, help='number of classes for the classification task')
    parser.add_argument('--tgt_test_address', default=None, type=str, help='address of image list of target domain dataset')


    # preprocessing
    parser.add_argument('--batch_size', default=50, type=int, help='batch size')
    parser.add_argument('--resize_size', default=256, type=int, help='resize original image to this size in dataloader')
    parser.add_argument('--crop_size', default=224, type=int, help='crop image to this size in dataloader')
    parser.add_argument('--crop_type', default='RandomResizedCrop', type=str, help='RandomResizedCrop, RandomCrop.')

    # implicit alignment
    parser.add_argument('--self_train',
                        action='store_true', help="whether to use self-training for sampling the target domain")
    parser.add_argument('--source_sample_mode',
                        action='store_true', help="whether to train the source domain with n way k shot sampled tasks")
    parser.add_argument('--n_way', default=40, type=int,
                        help='number of classes for each classification task')
    parser.add_argument('--k_shot', default=1, type=int,
                        help='number of examples per class per domain, default using all examples available.')
    parser.add_argument('--yhat_update_freq', type=int, default=20,
                        help='frequency to update the self-training predictions on the target domain')
    parser.add_argument('--self_train_sampler', type=str, default='SelfTrainingVannilaSampler',
                        help='sampler for self training, e.g., SelfTrainingVannilaSampler, SelfTrainingConfidentSampler')


    parser.add_argument('--confidence_threshold', type=float, default=None, help='threshold for conditional sampling')

    args = parser.parse_args()



    torch.random.manual_seed(100)

    data_loader_manager = load_data_and_train(args)

    totalNet = DSAN().cuda()
    soft_max = nn.Softmax(dim=1).cuda()
    log_soft_max = nn.LogSoftmax(dim=1).cuda()

    

    # optim
    lr = 0.001
    weight_decay = 0.0005
    momentum = 0.9
    optimizer = optim.SGD(totalNet.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    # data
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    source_train_ds = FileListDataset(list_path='officehome/txt/Real_World_RS.txt', path_prefix='officehome',
                                   transform=train_transform, filter=None)


    source_train_dl = DataLoader(dataset=source_train_ds, batch_size=32, shuffle=True,num_workers=4)


    source_model = SourceNet()
    
    source_model.load_state_dict(torch.load('/domain/pretrained//RwPre.pkl'))

    soft_labels = gen_soft_labels(65, source_train_dl, source_model)
    train(totalNet, data_loader_manager, soft_labels, optimizer=optimizer, args=args)

def gen_soft_labels(num_classes, src_train_loader, source_model):
    cuda = torch.cuda.is_available()
    temperature = 2
    soft_labels = torch.zeros(num_classes, 1, num_classes).cuda()
    sum_classes = torch.zeros(num_classes).cuda()
    pred_scores_total = []
    label_total = []
    if cuda:
        source_model = source_model.cuda()

    for _, (src_data, label) in enumerate(src_train_loader):
        label_total.append(label)
        if cuda:
            src_data, label = src_data.cuda(), label.cuda()
            src_data, label = Variable(src_data), Variable(label)
        output = source_model(src_data)

        pred_scores = F.softmax(output / temperature, dim=1).data.cuda()
        pred_scores_total.append(pred_scores)

    pred_scores_total = torch.cat(pred_scores_total)
    label_total = torch.cat(label_total)

    for i in range(len(src_train_loader.dataset)):
        sum_classes[label_total[i]] += 1
        soft_labels[label_total[i]][0] += pred_scores_total[i]
    for cl_idx in range(num_classes):
        soft_labels[cl_idx][0] /= sum_classes[cl_idx]
    return soft_labels
   


# soft label for each batch
def ret_soft_label(label, soft_labels):
    num_classes = 65
    soft_label_for_batch = torch.zeros(label.size(0), num_classes)
    for i in range(label.size(0)):
        soft_label_for_batch[i] = soft_labels[label.data[i]]

    return soft_label_for_batch


def evaluate_auc(data_iter, net):
    net.cuda()

    nc = 40
    ncArray = list(range(40))
    y_prob_sum = None
    y_sum = None
    with torch.no_grad():
        for _, X, y in data_iter:
            X = X.cuda()
            y = y.cuda()
            res = 0
            net.eval()
            y_hat, _ , _= net(X, X, y)
            y_source_prob = nn.Softmax(dim=1)(y_hat)
            if y_prob_sum == None:
                y_prob_sum = y_source_prob
                y_sum = y
            else:
                y_prob_sum = torch.cat([y_prob_sum, y_source_prob], dim=0)
                y_sum = torch.cat([y_sum, y], dim=0)
            net.train()

        y = y_sum
        for i in range(nc):
            index_i = y == i
            y_hat_i = y_prob_sum[index_i, i]
            ni = index_i.int().sum().item()
            if ni > 0:
                y_hat_i = y_hat_i.reshape(ni, 1)
                for otheri in ncArray:
                    if (otheri == i):
                        pass
                    else:
                        index_otheri = y == otheri
                        y_hat_otheri = y_prob_sum[index_otheri, i]
                        nj = index_otheri.int().sum().item()
                        if nj > 0:
                            y_hat_otheri = y_hat_otheri.reshape(nj, 1)
                            ninj = ni * nj
                            i_to_ninj = y_hat_i.expand(ni, nj)
                            y_hat_otheri_zz = y_hat_otheri.permute(1, 0)
                            otheri_to_ninj = y_hat_otheri_zz.expand(ni, nj)
                            good_pair_n = ((i_to_ninj - otheri_to_ninj) > 0).int().sum().item()
                            res += good_pair_n / ninj
        res = res / (nc * (nc - 1))
        print("auc:" + str(res))

    return res

def main():
    run()

if __name__ == '__main__':
    main()

    
