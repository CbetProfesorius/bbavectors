import torch
import torch.nn as nn
import os
import numpy as np
import loss
import cv2
import func_utils
import neptune

def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict

class TrainModule(object):
    def __init__(self, dataset, num_classes, model, decoder, down_ratio, down_ratio_kpts, kpts_radius, neptune_project=None, neptune_api=None):
        torch.manual_seed(317)
        self.dataset = dataset
        self.dataset_phase = {'TrimbleSlabs': ['train', 'val'],
                              'dota': ['train'],
                              'hrsc': ['train', 'test']}
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.decoder = decoder
        self.down_ratio = down_ratio
        self.down_ratio_kpts = down_ratio_kpts
        self.kpts_radius = kpts_radius
        self.neptune_project = neptune_project
        self.neptune_api = neptune_api

    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
        }, path)

    def load_model(self, model, optimizer, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return model, optimizer, epoch

    def train_network(self, args):
        if self.neptune_project:
            run = neptune.init_run(
                project=self.neptune_project, 
                api_token=self.neptune_api,
            )
        else:
            run = None
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)
        save_path = args.save_path+'_weights_'+args.dataset
        start_epoch = 1
        
        # add resume part for continuing training when break previously, 10-16-2020
        if args.resume_train:
            self.model, self.optimizer, start_epoch = self.load_model(self.model, 
                                                                        self.optimizer, 
                                                                        args.resume_train, 
                                                                        strict=True)
        # end 

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if args.ngpus>1:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        criterion = loss.LossAll()
        print('Setting up data...')

        dataset_module = self.dataset[args.dataset]

        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   down_ratio=self.down_ratio,
                                   down_ratio_kpts=self.down_ratio_kpts,
                                   kpts_radius=self.kpts_radius)
                 for x in self.dataset_phase[args.dataset]}

        dsets_loader = {}
        dsets_loader['train'] = torch.utils.data.DataLoader(dsets['train'],
                                                           batch_size=args.batch_size,
                                                           shuffle=True,
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           drop_last=True,
                                                           collate_fn=collater)
        
        dsets_loader['val'] = torch.utils.data.DataLoader(dsets['val'],
                                                           batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           drop_last=False,
                                                           collate_fn=collater)

        print('Starting training...')
        train_loss = {
            'total_loss': [],
            'hm_loss': [],
            'wh_loss': [],
            'off_loss': [],
            'cls_theta_loss': []
        }
        val_loss = {
            'total_loss': [],
            'hm_loss': [],
            'wh_loss': [],
            'off_loss': [],
            'cls_theta_loss': []
        }

        for epoch in range(start_epoch, args.num_epoch+1):
            print('-'*10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            train_epoch_total_loss, train_epoch_hm_loss, train_epoch_wh_loss, train_epoch_off_loss, train_epoch_cls_theta_loss, train_epoch_hm_kpts_loss, train_epoch_off_kpts_loss = self.run_epoch(phase='train',
                                                                                                                                                                                        data_loader=dsets_loader['train'],
                                                                                                                                                                                        criterion=criterion,
                                                                                                                                                                                        run=run)
            
            train_loss['total_loss'].append(train_epoch_total_loss)
            train_loss['hm_loss'].append(train_epoch_hm_loss)
            train_loss['wh_loss'].append(train_epoch_wh_loss)
            train_loss['off_loss'].append(train_epoch_off_loss)
            train_loss['cls_theta_loss'].append(train_epoch_cls_theta_loss)

            self.scheduler.step(epoch)

            val_epoch_total_loss, val_epoch_hm_loss, val_epoch_wh_loss, val_epoch_off_loss, val_epoch_cls_theta_loss, val_epoch_hm_kpts_loss, val_epoch_off_kpts_loss, epoch_kpts_metrics, epoch_kpts_mae = self.run_epoch(phase='val',
                                                                                                                                                                                                                        data_loader=dsets_loader['val'],
                                                                                                                                                                                                                        criterion=criterion,
                                                                                                                                                                                                                        run=run)

            val_loss['total_loss'].append(val_epoch_total_loss)
            val_loss['hm_loss'].append(val_epoch_hm_loss)
            val_loss['wh_loss'].append(val_epoch_wh_loss)
            val_loss['off_loss'].append(val_epoch_off_loss)
            val_loss['cls_theta_loss'].append(val_epoch_cls_theta_loss)

            mrec, mprec, map = self.dec_eval(args, dsets['val'])

            print('RBOX')
            print('mAP: {:.3f}'.format(map), 'Recall: {:.3f}'.format(mrec), 'Precision: {:.3f}'.format(mprec))

            print('Corners')
            for key, val in epoch_kpts_metrics.items():
                print(f'{key}: {val:.3f}', end=' ')
                
            if run:
                run['mAP'].log(map)
                run['Recall'].log(mrec)
                run['Precision'].log(mprec)
                for key, val in epoch_kpts_metrics.items():
                    run[f'{key}'].log(val)

            # if epoch % 5 == 0 or epoch > 20:
            self.save_model(os.path.join(save_path, 'model_{}.pth'.format(epoch)),
                            epoch,
                            self.model,
                            self.optimizer)

            # if 'test' in self.dataset_phase[args.dataset] and epoch%5==0:
            #     mAP = self.dec_eval(args, dsets['test'])
            #     ap_list.append(mAP)
                # np.savetxt(os.path.join(save_path, 'ap_list.txt'), ap_list, fmt='%.6f')

            self.save_model(os.path.join(save_path, 'model_last.pth'),
                            epoch,
                            self.model,
                            self.optimizer)
            
        if self.neptune_project:
            run.stop()

    def run_epoch(self, phase, data_loader, criterion, run=None):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
            
        running_total_loss = 0.
        running_hm_loss = 0.
        running_wh_loss = 0.
        running_off_loss = 0.
        running_cls_theta_loss = 0.
        running_hm_kpts_loss = 0.
        running_off_kpts_loss = 0.
        running_kpts_metrics = {}
        running_kpts_mae = 0.
        running_graph_loss = 0.
        running_edges_acc = {}

        for data_dict in data_loader:
            # print(f"Phase: {phase}, Input Shape: {data_dict['input'].shape}")
            for name in data_dict:
                data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs, graph_loss, graph_acc = self.model(data_dict['input'], data_dict, phase)
                    total_loss, hm_loss, wh_loss, off_loss, cls_theta_loss, hm_kpts_loss, off_kpts_loss, _, _ = criterion(pr_decs, data_dict)
                    total_loss += graph_loss
                    total_loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs, graph_loss, graph_acc = self.model(data_dict['input'], data_dict, phase)
                    total_loss, hm_loss, wh_loss, off_loss, cls_theta_loss, hm_kpts_loss, off_kpts_loss, kpts_metrics, kpts_mae = criterion(pr_decs, data_dict)
                    total_loss += graph_loss
                    # ap = self.dec_eval(args, pr_decs, dsets)
                   
                for key, value in kpts_metrics.items():
                    if key not in running_kpts_metrics:
                        running_kpts_metrics[key] = 0.
                    running_kpts_metrics[key] += value
                running_kpts_mae += kpts_mae


            running_total_loss += total_loss.item()
            running_hm_loss += hm_loss.item()
            running_wh_loss += wh_loss.item()
            running_off_loss += off_loss.item()
            running_cls_theta_loss += cls_theta_loss.item()
            running_hm_kpts_loss += hm_kpts_loss.item()
            running_off_kpts_loss += off_kpts_loss.item()
            running_graph_loss += graph_loss.item()
            
            for key, value in graph_acc.items():
                if key not in running_edges_acc:
                    running_edges_acc[key] = 0.
                running_edges_acc[key] += value
        
        epoch_total_loss = running_total_loss / len(data_loader)
        epoch_hm_loss = running_hm_loss / len(data_loader)
        epoch_wh_loss = running_wh_loss / len(data_loader)
        epoch_off_loss = running_off_loss / len(data_loader)
        epoch_cls_theta_loss = running_cls_theta_loss / len(data_loader)
        epoch_hm_kpts_loss = running_hm_kpts_loss / len(data_loader)
        epoch_off_kpts_loss = running_off_kpts_loss / len(data_loader)
        epoch_graph_loss = running_graph_loss / len(data_loader)

        epoch_edges_acc = {}
        for key, value in running_edges_acc.items():
            epoch_edges_acc[key] = value / len(data_loader)
        
        if phase == 'val':
            epoch_kpts_metrics = {}
            for key, value in running_kpts_metrics.items():
                epoch_kpts_metrics[key] = value / len(data_loader)
            epoch_kpts_mae = running_kpts_mae / len(data_loader)

        print('{} loss: {:.3f}'.format(phase, epoch_total_loss) + 
              ' hm_loss: {:.3f}'.format(epoch_hm_loss) + 
              ' wh_loss: {:.3f}'.format(epoch_wh_loss) + 
              ' off_loss: {:.3f}'.format(epoch_off_loss) + 
              ' cls_theta_loss: {:.3f}'.format(epoch_cls_theta_loss) +
              ' hm_kpts_loss: {:.3f}'.format(epoch_hm_kpts_loss) + 
              ' off_kpts_loss: {:.3f}'.format(epoch_off_kpts_loss) + 
              ' graph_loss: {:.3f}'.format(epoch_graph_loss)
              )
        
        print('Edges')
        for key, value in epoch_edges_acc.items():
            print(f'{key}: {value:.3f}', end=' ')
        
        if run:
            run[f"{phase}_total_loss"].log(epoch_total_loss)
            run[f"{phase}_hm_loss"].log(epoch_hm_loss)
            run[f"{phase}_wh_loss"].log(epoch_wh_loss)
            run[f"{phase}_off_loss"].log(epoch_off_loss)
            run[f"{phase}_cls_theta_loss"].log(epoch_cls_theta_loss)
            run[f"{phase}_hm_kpts_loss"].log(epoch_hm_kpts_loss)
            run[f"{phase}_off_kpts_loss"].log(epoch_off_kpts_loss)
            run[f"{phase}_graph_loss"].log(epoch_graph_loss)

            for key, value in epoch_edges_acc.items():
                run[f"{phase}_{key}"].log(value)
        
        if phase == 'val':
            return epoch_total_loss, epoch_hm_loss, epoch_wh_loss, epoch_off_loss, epoch_cls_theta_loss, epoch_hm_kpts_loss, epoch_off_kpts_loss, epoch_kpts_metrics, epoch_kpts_mae
        else:
            return epoch_total_loss, epoch_hm_loss, epoch_wh_loss, epoch_off_loss, epoch_cls_theta_loss, epoch_hm_kpts_loss, epoch_off_kpts_loss

    def dec_eval(self, args, dsets):
        # result_path = 'result_'+args.dataset
        # if not os.path.exists(result_path):
        #     os.mkdir(result_path)

        self.model.eval()

        results = func_utils.write_results(args,
                                            self.model,
                                            dsets,
                                            self.down_ratio,
                                            self.down_ratio_kpts,
                                            self.device,
                                            self.decoder)
        
        mrec, mprec, map = dsets.dec_evaluation(results, args.conf_thresh)
        # , kpts_prec, kpts_rec, kpts_f1 = dsets.dec_evaluation(results, args.conf_thresh)
        return float(mrec), float(mprec), float(map)
    # , kpts_prec, kpts_rec, kpts_f1