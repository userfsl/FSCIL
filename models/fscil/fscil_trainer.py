from models.base.fscil_trainer import FSCILTrainer as Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *
from .Network import MYNET


import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import itertools


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        # self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels, matrix):
        for p, t in zip(preds, labels):
            # self.matrix[p, t] += 1
            matrix[p, t] += 1
        return matrix

    def plot(self, session, matrix, normalize=True):
        # cm = self.matrix
        cm = matrix
        classes = []
        for i in range(self.num_classes):
            classes.append(i)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
            print(cm)
        else:
            print(cm)
        
        # matrix = self.matrix
        # plt.imshow(matrix , interpolation='nearest', cmap=cmap)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.imshow(cm, interpolation='nearest')
        title='Confusion matrix'
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        a = [0,20,40,60,80,100,120,140,160,180,200]
        # labels = ['A', 'B', 'C', 'D','E']
        plt.xticks(a, a, rotation = 30)
        plt.yticks(a, a, rotation = 30)
        # plt.xticks(tick_marks, classes, rotation=45)
        # plt.yticks(tick_marks, classes)
        # plt.ylim(len(classes) - 0.5, -0.5)
        # fmt = '.2f' if normalize else 'd'
        fmt = '.2f'# if normalize else 'd'
        thresh = cm.max() / 2.
        # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #     plt.text(j, i, format(cm[i, j], fmt),
        #              horizontalalignment="center",
        #              color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        plt.savefig('%s.png'%(session), dpi=1600)
        plt.close()



class FSCILTrainer(Trainer):
    def __init__(self, args):

        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.set_up_model()
        pass

    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.base_mode)
        print(MYNET)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir != None:  #
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('*********WARNINGl: NO INIT MODEL**********')
            # raise ValueError('You must initialize a pre-trained model')
            pass

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = self.get_base_dataloader_meta()
        else:
            trainset, trainloader, testloader = self.get_new_dataloader(session)
        return trainset, trainloader, testloader

    def get_base_dataloader_meta(self):
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(0 + 1) + '.txt'
        class_index = np.arange(self.args.base_class)
        if self.args.dataset == 'cifar100':
            # class_index = np.arange(self.args.base_class)
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=True,
                                                  index=class_index, base_sess=True)
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_index, base_sess=True)

        if self.args.dataset == 'cub200':
            # class_index = np.arange(self.args.base_class)
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_index)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_index)
        # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        sampler = CategoriesSampler(trainset.targets, self.args.train_episode, self.args.episode_way, 300) # 30 for CUB dataset, 300 for miniImageNet
                                    # self.args.episode_shot + self.args.episode_query)

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=8,
                                                  pin_memory=True)

        testloader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

        return trainset, trainloader, testloader

    def get_new_dataloader(self, session):
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(session + 1) + '.txt'
        if self.args.dataset == 'cifar100':
            class_index = open(txt_path).read().splitlines()
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=False,
                                                  index=class_index, base_sess=False)
        if self.args.dataset == 'cub200':
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True,
                                                index_path=txt_path)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True,
                                                      index_path=txt_path)
        if self.args.batch_size_new == 0:
            batch_size_new = trainset.__len__()
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                      num_workers=8, pin_memory=True)
        else:
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.args.batch_size_new,
                                                      shuffle=True,
                                                      num_workers=8, pin_memory=True)

        class_new = self.get_session_classes(session)

        if self.args.dataset == 'cifar100':
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_new, base_sess=False)
        if self.args.dataset == 'cub200':
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_new)
        if self.args.dataset == 'mini_imagenet':
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_new)

        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=self.args.test_batch_size, shuffle=False,
                                                 num_workers=8, pin_memory=True)

        return trainset, trainloader, testloader

    def get_session_classes(self, session):
        class_list = np.arange(self.args.base_class + session * self.args.way)
        return class_list

    def replace_to_rotate(self, proto_tmp, query_tmp):
        for i in range(self.args.low_way):
            # random choose rotate degree
            rot_list = [90, 180, 270]
            sel_rot = random.choice(rot_list)
            if sel_rot == 90:  # rotate 90 degree
                # print('rotate 90 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
            elif sel_rot == 180:  # rotate 180 degree
                # print('rotate 180 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].flip(2).flip(3)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].flip(2).flip(3)
            elif sel_rot == 270:  # rotate 270 degree
                # print('rotate 270 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
        return proto_tmp, query_tmp

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                     {'params': self.model.module.slf_attn.parameters(), 'lr': self.args.lrg},
                                     {'params': self.model.module.slf_attn_p.parameters(), 'lr': self.args.lr_new},
                                     {'params': self.model.module.simcom.parameters(), 'lr': self.args.lrg}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            self.model = self.update_param(self.model, self.best_model_dict)

            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                
                start_time = time.time()
                # train base sess
                self.model.eval()
                tl, ta, query_val, semantic, proto, label_val = self.base_train(self.model, trainloader, optimizer, scheduler, args)

                self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)

                self.model.module.mode = 'avg_cos'

                if args.set_no_val: # set no validation
                    save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    torch.save(dict(params=self.model.state_dict()), save_model_dir)
                    torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    tsl, tsa = self.test(self.model, testloader, args, session)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    print('lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                        lrc, tl, ta, tsl, tsa))
                    result_list.append(
                        'lr:%.5f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            lrc, tl, ta, tsl, tsa))
                else:
                    # take the last session's testloader for validation
                    # vl, va = self.validation1(self.model, query_val, semantic, proto, label_val, args)
                    # vl, va = self.validation()
                    vl, va = self.test(self.model, testloader, args, session)
                    print(va)
                    vl0, va0 = self.validation()

                    # save better model
                    if (va * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                        # self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    # print('best epoch {}, best val acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                      # self.trlog['max_acc'][session]))
                    self.trlog['val_loss'].append(vl)
                    self.trlog['val_acc'].append(va)
                    lrc = scheduler.get_last_lr()[0]
                    print('lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                        lrc, tl, ta, vl, va))
                    result_list.append(
                        'lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                            lrc, tl, ta, vl, va))

                self.trlog['train_loss'].append(tl)
                self.trlog['train_acc'].append(ta)

                scheduler.step()


                # always replace fc with avg mean
                self.model.load_state_dict(self.best_model_dict)
                self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                torch.save(dict(params=self.model.state_dict()), best_model_dir)

                self.model.module.mode = 'avg_cos'
                tsl, tsa = self.test(self.model, testloader, args, session)
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                print('The test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                # result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    # session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))



            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                self.model.load_state_dict(self.best_model_dict)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                tsl, tsa = self.test(self.model, testloader, args, session)

                # save better model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))

                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                # result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    # session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        # result_list.append('Best epoch:%d' % self.trlog['max_acc_epoch'])
        # print('Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

    def validation1(self, model, query, semantic, proto, test_label, args):
        # self.model.module.mode = self.args.new_mode
        test_class = args.episode_way#base_class
        model = model.eval()
        vl = Averager()
        va = Averager()
        with torch.no_grad():
            model.module.mode = 'encoder'
            query = model(query)
            query = query.view(int(query.size(0)/test_class), test_class, query.size(-1))
            query = query.unsqueeze(0)#.unsqueeze(0)

            # proto = model.module.fc.weight[:test_class, :].detach()
            # proto = proto.unsqueeze(0).unsqueeze(0)

            logits = model.module._forward(proto, query, semantic, False, proto, proto)
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

        return loss, acc


    def validation(self):
        with torch.no_grad():
            model = self.model
            accuracy_list = []
            loss_list = []
            for session in range(1, self.args.sessions):
                train_set, trainloader, testloader = self.get_dataloader(session)

                trainloader.dataset.transform = testloader.dataset.transform
                model.module.mode = 'avg_cos'
                model.eval()
                model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                vl, va = self.test(model, testloader, self.args, session)
                accuracy_list.append(va)
                loss_list.append(vl)
        print(accuracy_list)

        return loss_list[0], accuracy_list[0]

    def base_train(self, model, trainloader, optimizer, scheduler,  args):
        tl = Averager()
        ta = Averager()

        # tqdm_gen = tqdm(trainloader)

        # label = torch.arange(args.episode_way + args.low_way).repeat(args.episode_query)
        label = torch.arange(args.episode_way).repeat(args.episode_query)
        label = label.type(torch.cuda.LongTensor)

        for i, batch in enumerate(trainloader, 1):
            # data, true_label, attributes = [_.cuda() for _ in batch]
            data, true_label, attributes = [_.cuda() for _ in batch]


            k = args.episode_way * args.episode_shot
            # proto, query = data[:k], data[k:]

            model.module.mode = 'encoder'
            # data = model(data)

            # proto_tmp = model(proto_tmp)
            # query_tmp = model(query_tmp)

            k_1 = args.episode_way * args.episode_query
            # proto, query = data[:k], data[k:]
            proto = data[:k]
            query = data[-k_1:]
            query_val = data[k:-k_1]
            proto = model(proto)
            query = model(query)
            semantic = attributes[:k]

            label_val = torch.arange(args.episode_way).repeat(int(query_val.size(0)/args.episode_way)).cuda()
            # query_val = model()

            proto = proto.view(args.episode_shot, args.episode_way, proto.shape[-1])

            for num in range(args.episode_way):
                var_ind = 0.0
                for index in range(proto.size(0)):
                    x_i = proto[index,num,:]-proto[:,num,:].mean(0)
                    x_i = x_i.unsqueeze(0)
                    var_ind = var_ind + torch.matmul(x_i.T, x_i)
                # var_ind = torch.cov(proto[:,num,:].T) 
                var_ind = var_ind/(proto.size(0)-1)
                if num == 0:
                    var_list = var_ind.unsqueeze(0)
                else:
                    var_list =  torch.cat([var_list, var_ind.unsqueeze(0)] , dim = 0)

            query = query.view(args.episode_query, args.episode_way, query.shape[-1])
            semantic = semantic.view(args.episode_shot, args.episode_way, semantic.shape[-1])

            random_label = []
            lamb = 0.5
            for num in range(args.low_way):
                random_mix = random.sample(range(0, args.episode_way), 2)
                # random_label.append(random_mix[0])
                proto_mix = lamb * proto[:,random_mix[0],:].mean(0) + (1 - lamb) * proto[:,random_mix[-1],:].mean(0)
                semantic_mix = lamb * semantic[:,random_mix[0],:].mean(0) + (1 - lamb) * semantic[:,random_mix[-1],:].mean(0)
                cov_mix = lamb * var_list[random_mix[0],:,:] + (1 - lamb) * var_list[random_mix[-1],:,:]

                if num == 0:
                    cov_list = cov_mix.unsqueeze(0)
                else:
                    cov_list = torch.cat([cov_list, cov_mix.unsqueeze(0)], dim=0)

        
                proto_final = proto_mix
                if num == 0:
                    proto_bg = proto_mix.unsqueeze(0)
                    # proto_list = proto_final
                    proto_list = proto_final.unsqueeze(0)
                    semantic_list = semantic_mix.unsqueeze(0)
                else:
                    proto_bg = torch.cat([proto_bg, proto_mix.unsqueeze(0)], 0)
                    # proto_list = torch.cat([proto_list, proto_final], 0)
                    proto_list = torch.cat([proto_list, proto_final.unsqueeze(0)], 0)
                    semantic_list = torch.cat([semantic_list, semantic_mix.unsqueeze(0)], 0)
            proto = proto.mean(0)#.unsqueeze(0)
            if args.low_shot > 0:
                # proto = torch.cat([proto, proto_list.unsqueeze(0)], 0)
                # proto = torch.cat([proto, proto_list], 0) 
                proto = torch.cat([proto, proto_list.squeeze(1)], 0)
            
            # proto[:args.low_way] = proto_tmp # new added few-shot scenerio
            # proto[random_list] = proto_tmp
            proto = proto.unsqueeze(0)
            semantic = semantic.mean(0).unsqueeze(0)
            if args.low_way > 0:
                semantic = torch.cat([semantic, semantic_list.unsqueeze(0)], 1)

            # proto = torch.cat([proto, proto_tmp], dim=1)
            # query = torch.cat([query, query_tmp], dim=1)

            proto = proto.unsqueeze(0)
            query = query.unsqueeze(0)


            if args.low_way == 0:
                cov_list = var_list

            logits = model.module._forward(proto, query, semantic, True, var_list, cov_list)


            label = label.view(args.episode_query, args.episode_way)
            label_new = torch.arange(args.episode_way, args.episode_way + args.low_way).repeat(args.episode_query) #
            label_new = label_new.view(args.episode_query, args.low_way).cuda()
            label_new = label_new[:, :int(args.low_way/4)]
            label_new = torch.cat([label, label_new], dim = -1)
            label_new = label_new.view(-1)
            if args.low_way == 0:
                label_new = label.view(-1)


            total_loss = F.cross_entropy(logits, label_new) #+ 1.0 * loss_sim + 1.0 * uniformity# + loss_sim
            if args.low_way > 0:
                total_loss = total_loss
 

            acc = count_acc(logits, label_new)

            lrc = scheduler.get_last_lr()[0]
            print('Session 0, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        tl = tl.item()
        ta = ta.item()

        return tl, ta, query_val, semantic[:,:args.episode_way], proto[:,:,:args.episode_way,:], label_val.view(-1)

    def test(self, model, testloader, args, session):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        matrix = np.zeros((test_class, test_class))
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                _, _, attributes = [_.cuda() for _ in batch]
                if i == 1:
                    semantic = attributes
                semantic = torch.cat([semantic, attributes], dim=0)
                semantic = torch.unique(semantic, dim=0)
            for i, batch in enumerate(testloader, 1):
                data, test_label, _ = [_.cuda() for _ in batch]
                # semantic = attributes.view(-1, test_class, attributes.shape[-1]).mean(0).unsqueeze(0)

                model.module.mode = 'encoder'
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)

                proto = model.module.fc.weight[:test_class, :].detach()
                proto = proto.unsqueeze(0).unsqueeze(0)

                logits = model.module._forward(proto, query, semantic.unsqueeze(0), False, proto, proto)

                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()
        # confusion.plot(session, matrix)
        return vl, va

    def set_save_path(self):

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project
        self.args.save_path = self.args.save_path + '%dW-%dS-%dQ-L%dW-L%dS' % (
            self.args.episode_way, self.args.episode_shot, self.args.episode_query,
            self.args.low_way, self.args.low_shot)
        # if self.args.use_euclidean:
        #     self.args.save_path = self.args.save_path + '_L2/'
        # else:
        #     self.args.save_path = self.args.save_path + '_cos/'
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Lr1_%.6f-Lrg_%.5f-MS_%s-Gam_%.2f-T_%.2f' % (
                self.args.lr_base, self.args.lrg, mile_stone, self.args.gamma,
                self.args.temperature)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Lr1_%.6f-Lrg_%.5f-Step_%d-Gam_%.2f-T_%.2f' % (
                self.args.lr_base, self.args.lrg, self.args.step, self.args.gamma,
                self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
