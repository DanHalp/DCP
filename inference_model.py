# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime
import os
from easydict import EasyDict as edict
from pathlib import Path
import torch
from SpareNet.datasets.data_loaders import ShapeNetDataLoader, DatasetSubset, collate_fn
import numpy as np
from scipy.spatial.transform import Rotation
from DCP.model import DCP
from DCP.util import transform_point_cloud, npmat2euler
import torch.nn.functional as F
from Configs.pr_config import *
from DCP.data import ModelNet40
import torch
from tqdm import tqdm
import time
from SpareNet.Inference_model import SpareNet
from torch.utils.tensorboard import SummaryWriter



class DCP_MODEL():
    """
    This model warps the entire pipline for applying affine transformations, using the DCP, on the ShapeNet dataset.
    One could test on three differnt point clouds types: 
                ground truth ("gt", complete objects)
                partial
                reconstructed (By using a pretrain model of SpareNet, by microsoft.)
    """
    
    def __init__(self, args, gaussian_noise=False, partiton=TASK.TEST) -> None:
        
        self.gaussian_noise = gaussian_noise
        self.partition = partiton
        self.device = args.PROJECT.device
        self._model, self._cfg  = self.make_model(args)
        
        if self._cfg.RECONSTRUCTION.active:
            self.init_reconstructor()
        
        if partiton==TASK.TEST:
            self._model.eval()
            
            if  self._cfg.TRANSFORM.dataset == DATASETS.SHAPENET:
                self.dataset = ShapeNetDataLoader(args).get_dataset(DatasetSubset.TEST)
                self._cfg.TRANSFORM.fixed_dataset_indices = np.arange(20)
                if self._cfg.TRANSFORM.fixed_dataset_indices is not None:
                    indi = self._cfg.TRANSFORM.fixed_dataset_indices
                else:
                    indi = np.random.permutation(np.arange(len(self.dataset)))[:self._cfg.TRANSFORM.test_batch_size]
                self.dataset = [self.dataset[i] for i in indi]
                
            elif self._cfg.TRANSFORM.dataset == DATASETS.MODELENT:
                self.dataset  = ModelNet40(num_points=self._cfg.TRANSFORM.num_points, partition='test', gaussian_noise=self._cfg.TRANSFORM.gaussian_noise,
                            unseen=self._cfg.TRANSFORM.unseen, factor=self._cfg.TRANSFORM.factor)
            else:
                raise Exception("An invalid dataset name was given. Check the relevant variable in pr_config.py")
             
        # We train on shapenet, since we already have a pretrained model for Modelnet.
        else:
            self._model.train()

            p = self._cfg.TRANSFORM.dataset_fraction
            self.train_dataset =  ShapeNetDataLoader(args).get_dataset(DatasetSubset.TRAIN)
            self.train_dataset.file_list = self.train_dataset.file_list[:int(len(self.train_dataset.file_list) * p)]
            self.train_data_loader = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self._cfg.TRAIN.batch_size,
                num_workers=self._cfg.CONST.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                shuffle=True,
                drop_last=True,
            )
            self.test_dataset =  ShapeNetDataLoader(args).get_dataset(DatasetSubset.TEST)
            self.test_dataset.file_list = self.test_dataset.file_list[:int(len(self.test_dataset.file_list) * p)]
            self.test_data_loader = torch.utils.data.DataLoader(
                dataset=self.test_dataset,
                batch_size=1,
                num_workers=self._cfg.CONST.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                shuffle=False,
            )
            self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self._cfg.TRANSFORM.lr, weight_decay=1e-4)
            # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150, 200], gamma=0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self._cfg.TRANSFORM.optim_scheduler_step_size, 
                                                             gamma=self._cfg.TRANSFORM.optim_scheduler_gamma)
            
            self._cfg.TRANSFORM.checkpoints_path = Path(f"Checkpoints/{self._cfg.TRANSFORM.dataset_name}") / datetime.now().strftime('run_%H_%M_%d_%m_%Y')
           
            os.makedirs(self._cfg.TRANSFORM.checkpoints_path, exist_ok=True)
            self.tblog = SummaryWriter(self._cfg.TRANSFORM.checkpoints_path / "logs")
            
                    
    def init_reconstructor(self):
        self.reconstructor = SpareNet(self._cfg)
    
    def make_model(self, args):
        
        assert "TRANSFORM" in args, "Wrong cfg, ya habibi"

        args.PROJECT.update(args.RECONSTRUCTION) # Important for the dataloader, as it is initially a part of the reconstruction repo.
        args.CONST.num_workers = 2 # Suggested by pytorch
        model = DCP(args).to(self.device)
        
        # Load pretrained weights from ModelNet
        if args.TRANSFORM.model_path != "":
            model.load_state_dict(torch.load(args.TRANSFORM.model_path), strict=False)
            print(f"Loaded the weights from {args.TRANSFORM.model_path}")

        return model, args
    
    def create_random_trans(self, pc, gt):
        
        old_shape = pc.size()
        if len(old_shape) > 2:
            pc = np.squeeze(pc)
            gt = np.squeeze(gt)
            
        if pc.shape[0] != 3:
            pc = pc.T
            gt = gt.T
            assert pc.shape[0] == 3 and gt.shape[0] == 3, "Your point cloud should be of shape (3xPointNum)"
            
        pointcloud1 = pc[:, :self._cfg.TRANSFORM.num_points]
        pointcloud2 = gt[:, :self._cfg.TRANSFORM.num_points]
        # pointcloud2 = pointcloud1
        # indi = np.random.permutation(np.arange(pc.shape[1]))[:self._cfg.TRANSFORM.num_points]
        # pointcloud1 = pc[:, indi]
        # indi = np.random.permutation(np.arange(gt.shape[1]))[:self._cfg.TRANSFORM.num_points]
        # pointcloud2 = gt[:, indi]

        if self.gaussian_noise or CFG.PROJECT.task  == TASK.TRAIN:
            pointcloud1 = self.jitter_pointcloud(pointcloud1)
        if self.partition != TASK.TRAIN:
            np.random.seed(np.random.randint(len(self.dataset)))
            
        anglex = np.random.uniform() * np.pi / self._cfg.TRANSFORM.factor
        angley = np.random.uniform() * np.pi / self._cfg.TRANSFORM.factor
        anglez = np.random.uniform() * np.pi / self._cfg.TRANSFORM.factor

        # cosx = np.cos(anglex)
        # cosy = np.cos(angley)
        # cosz = np.cos(anglez)
        # sinx = np.sin(anglex)
        # siny = np.sin(angley)
        # sinz = np.sin(anglez)
        # Rx = np.array([[1, 0, 0],
        #                 [0, cosx, -sinx],
        #                 [0, sinx, cosx]])
        # Ry = np.array([[cosy, 0, siny],
        #                 [0, 1, 0],
        #                 [-siny, 0, cosy]])
        # Rz = np.array([[cosz, -sinz, 0],
        #                 [sinz, cosz, 0],
        #                 [0, 0, 1]])
        # R_ab = Rx.dot(Ry).dot(Rz)
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])

        # R_ab = Rx.dot(Ry).dot(Rz)
        R_ab = rotation_ab.as_matrix()
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud2.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = torch.from_numpy(np.random.permutation(pointcloud1.T).T.astype('float32')).to(self.device)
        pointcloud2 = torch.from_numpy(np.random.permutation(pointcloud2.T).T.astype('float32')).to(self.device)
        R_ab = torch.from_numpy(R_ab.astype('float32')).to(self.device)
        translation_ab = torch.from_numpy(translation_ab.astype('float32')).to(self.device)
        R_ba = torch.from_numpy(R_ba.astype('float32')).to(self.device)
        translation_ba = torch.from_numpy(translation_ba.astype('float32')).to(self.device)
        euler_ab = torch.from_numpy(euler_ab.astype('float32')).to(self.device)
        euler_ba = torch.from_numpy(euler_ba.astype('float32')).to(self.device)
        
        if len(old_shape) == 3:
            pointcloud1 = pointcloud1[None, :]
            pointcloud2 = pointcloud2[None, :]

        # indi = np.random.permutation(np.arange(pc.shape[1]))[:self._cfg.TRANSFORM.num_points]
        # pointcloud1 = pc[:, indi]
        # indi = np.random.permutation(np.arange(gt.shape[1]))[:self._cfg.TRANSFORM.num_points]
        # pointcloud2 = gt[:, indi]
        return pointcloud1, pointcloud2, R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba
            
        
    def jitter_pointcloud(self, pointcloud, sigma=0.01, clip=0.05):
        N, C = pointcloud.shape
        pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        return pointcloud
    
    def sample_pc(self, ind=-1):
        
        
        if self._cfg.TRANSFORM.dataset == DATASETS.SHAPENET:
            if ind < 0:
                ind =  np.random.choice(np.arange(len(self.dataset)))
            gt = self.dataset[ind][-1][PC_TYPES_LIST[PC_TYPES.GT_CLOUD]].to(self.device).reshape(1, -1, 3)
            x = self.dataset[ind][-1][self._cfg.TRANSFORM.pc_type].to(self.device).reshape(1, -1, 3)
            
    
            if self._cfg.RECONSTRUCTION.active:
                assert self._cfg.TRANSFORM.pc_type in [PC_TYPES_LIST[PC_TYPES.PARTIAL_CLOUD],PC_TYPES_LIST[PC_TYPES.GT_CLOUD]], \
                    f"Only partial clouds at gtcloud can be reconstructed, not {self._cfg.TRANSFORM.pc_type}"
                
                data = {PC_TYPES_LIST[PC_TYPES.PARTIAL_CLOUD]: x, PC_TYPES_LIST[PC_TYPES.GT_CLOUD]: gt}
                _, _, refine_ptcloud, _ = self.reconstructor.complete(data)
                # dis = self.reconstructor.chamf_dis(refine_ptcloud, gt)
                # indi = torch.where(dis[0].squeeze() <= dis[0].mean())[0]
                # x = refine_ptcloud[:, indi].reshape(1, -1, 3)
                x = refine_ptcloud
                print(f"######################## Reconstructed pc {ind}")
                           
        else:
            if ind < 0:
                ind =  np.random.choice(np.arange(len(self.dataset.data)))
            x = torch.from_numpy(self.dataset.data[ind]).to(self.device).reshape(1, -1, 3)
            gt = x

        return x, gt
    
    def estimate_rotation(self, x, y):
        return self._model(x.to(self.device), y.to(self.device))
        

    
    @property
    def model(self):
        return self._model
    
    @property
    def cfg(self):
        return self._cfg

    def test(self):
        self.model.test()
    
    def create_log_files(self, cloud_type):
        
        pc_path = Path(CFG.TRANSFORM.output) / CFG.TRANSFORM.dataset_name / cloud_type
        make_dir(pc_path)
        metrics_path = pc_path  / "Metrics"
        make_dir(metrics_path)
        self._cfg.TRANSFORM.current_out_dir = pc_path
        self._cfg.TRANSFORM.metrics_path = metrics_path
        
        time = datetime.now().strftime('run_%H_%M_%d_%m_%Y')
        metrics_file = IOStream(self._cfg.TRANSFORM.metrics_path / f"metrics_{self._cfg.TRANSFORM.dataset_name}_{cloud_type}_{time}.csv")
        metrics_file.write_file("Iteration,ID,Loss,Cycle_Loss,MSE,RMSE,MAE,rot_MSE,rot_RMSE,rot_MAE,trans_MSE,trans_RMSE,trans_MAE")
        xxx_file = IOStream(self._cfg.TRANSFORM.metrics_path / f"transformation_{self._cfg.TRANSFORM.dataset_name}_{cloud_type}_{time}.csv")
        xxx_file.write_file("Sample_ID;type;rotation_ab;translation_ab;rotation_ba;translation_ba")
    
        return xxx_file, metrics_file
    
    def compare_all(self):
        '''
        This function compares the partial, reconstructed and the gt rotations, according to the same fixed rotation (Randomly sampled at the begining),
        for fair comparison
        '''
        
        # Estimate the initial rotation on the ground_truth.
        self._cfg.RECONSTRUCTION.active = False
        self._cfg.TRANSFORM.pc_type = PC_TYPES_LIST[PC_TYPES.GT_CLOUD]
        batch = list(self.create_batch())
        loss = self.test_and_print(self._cfg.TRANSFORM.pc_type, pre_defined_batch=batch)
        print(f"Evaluated the transformation for the ground truths, with loss: {loss}")

        # Calculate the partial point cloud.
        self._cfg.TRANSFORM.pc_type = PC_TYPES_LIST[PC_TYPES.PARTIAL_CLOUD]
        src, _, _, _, _, _, _, _ = self.create_batch(rotation_is_known=True)
        batch[0] = src
        loss =self.test_and_print(self._cfg.TRANSFORM.pc_type, pre_defined_batch=batch)
        print(f"Evaluated the transformation for the partial clouds, with loss {loss}")
        
        # Evaluate reconstrucion.
        self._cfg.RECONSTRUCTION.active = True
        self.init_reconstructor()
        src, _, _, _, _, _, _, _ = self.create_batch(rotation_is_known=True)
        batch[0] = src
        loss = self.test_and_print(PC_TYPES_LIST[PC_TYPES.RECONSTRUCTED_CLOUD], pre_defined_batch=batch)
        print(f"Evaluated the transformation for the reconstructed clouds, with loss {loss}")
        
    def create_batch(self, rotation_is_known=False):
        
        if self._cfg.TRANSFORM.dataset == DATASETS.MODELNET:
            size = min(len(self.dataset.data), self._cfg.TRANSFORM.test_batch_size)
        else:
            size = min(len(self.dataset), self._cfg.TRANSFORM.test_batch_size)
        rotations_ab = []
        translations_ab = []

        rotations_ba = []
        translations_ba = []

        eulers_ab = []
        eulers_ba = []
        
        srcs = []
        targets = []
        
        
        for i in range(size):
           
            pc, gt = self.sample_pc(i)
            pc = pc.transpose(2,1)
            
            gt = gt.transpose(2,1)
            
            src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba = self.create_random_trans(pc.cpu(), gt.cpu())
            
            if rotation_is_known:
                srcs.append(src)
                continue
            
            rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba = torch.unsqueeze(rotation_ab, dim=0), torch.unsqueeze(translation_ab, dim=0), \
                                                                                                    torch.unsqueeze(rotation_ba, dim=0), torch.unsqueeze(translation_ba, dim=0), \
                                                                                                    torch.unsqueeze(euler_ab, dim=0), torch.unsqueeze(euler_ba, dim=0)
            # Transform gt with random transformation
            
            # gt = transform_point_cloud(gt, rotation_ab, translation_ab)
            srcs.append(src)
            targets.append(target)
            

            ## save rotation and translation
            rotations_ab.append(rotation_ab)
            translations_ab.append(translation_ab)
            eulers_ab.append(euler_ab.detach().cpu().numpy())
            ##
            rotations_ba.append(rotation_ba)
            translations_ba.append(translation_ba)
            eulers_ba.append(euler_ba.detach().cpu().numpy())
        
        if rotation_is_known:
            srcs = torch.cat(srcs, axis=0)
            return srcs, targets, rotations_ab, translations_ab, eulers_ab, rotations_ba, translations_ba, eulers_ba
        
        srcs = torch.cat(srcs, axis=0).to(self.device)
        targets = torch.cat(targets, axis=0).to(self.device)

        rotations_ab = torch.cat(rotations_ab, axis=0).to(self.device)
        translations_ab = torch.cat(translations_ab, axis=0).to(self.device)
        eulers_ab = np.concatenate(eulers_ab, axis=0)

        rotations_ba = torch.cat(rotations_ba, axis=0).to(self.device)
        translations_ba = torch.cat(translations_ba, axis=0).to(self.device)
        eulers_ba = np.concatenate(eulers_ba, axis=0)
        
        return srcs, targets, rotations_ab, translations_ab, eulers_ab, rotations_ba, translations_ba, eulers_ba

    def test_and_print(self, cloud_type, pre_defined_batch=None):

        output, xxx_file = self.create_log_files(cloud_type)
        if pre_defined_batch is None:
            src, target, rotation_ab, translation_ab, euler_ab, rotation_ba, translation_ba, euler_ba = self.create_batch()
        else:
            assert len(pre_defined_batch) == 8, "A batch should be a predefined batch, containing [src, target, rotation_ab, translation_ab, euler_ab, rotation_ba, translation_ba, euler_ba]"
            src, target, rotation_ab, translation_ab, euler_ab, rotation_ba, translation_ba, euler_ba = pre_defined_batch
        

        for i in range(rotation_ab.shape[0]):
            xxx_file.write_file('%i;%s;%s;%s;%s;%s'
                % (i,"ground_truth",str(rotation_ab[i].tolist()), str(translation_ab[i].tolist()), str(rotation_ba[i].tolist()), str(translation_ba[i].tolist())))

        translation_ab_pred_acc, translation_ba_pred_acc = torch.zeros(translation_ab.shape, dtype=torch.float32).cuda(), torch.zeros(translation_ba.shape, dtype=torch.float32).cuda()
        rotation_ab_pred_acc = torch.eye(3).cuda().unsqueeze(0).repeat(src.shape[0], 1, 1).cuda()
        rotation_ba_pred_acc = torch.eye(3).cuda().unsqueeze(0).repeat(src.shape[0], 1, 1).cuda()
        src_acc = torch.clone(src)
        
        for i in range(src.shape[0]):
            output_folder = os.path.join(self._cfg.TRANSFORM.current_out_dir, str(i))
            os.makedirs(
                output_folder,
                exist_ok=True,
            )
            write_pcd_to_obj(os.path.join(output_folder, "src.obj"), src[i].squeeze().detach().cpu().numpy().T, c=(1, 0, 0))
            write_pcd_to_obj(os.path.join(output_folder, "target.obj"), target[i].squeeze().detach().cpu().numpy().T, c=(0, 1, 1))
        
        ret_loss = []
        for iteration in range(self._cfg.TRANSFORM.iterations):
            test_loss, test_cycle_loss, \
            test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, \
            rotation_ab_pred_acc, translation_ab_pred_acc, \
            rotation_ba_pred_acc, translation_ba_pred_acc, \
            src_acc = self.test_one_iteration(src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, \
                                    rotation_ab_pred_acc, translation_ab_pred_acc, \
                                    rotation_ba_pred_acc, translation_ba_pred_acc, src_acc, iteration)
            
            for i in range(rotation_ab.shape[0]):
                xxx_file.write_file('%i;%s;%s;%s;%s;%s'
                    % (i,"pred_" + str(iteration),str(rotation_ab_pred_acc[i].tolist()), str(translation_ab_pred_acc[i].tolist()), str(rotation_ba_pred_acc[i].tolist()), str(translation_ba_pred_acc[i].tolist())))
            
            test_rmse_ab = np.sqrt(test_mse_ab)
            test_rmse_ba = np.sqrt(test_mse_ba)

            test_rotations_ab_pred_euler = npmat2euler(rotation_ab_pred_acc.detach().cpu().numpy())
            test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(euler_ab)) ** 2)
            test_r_rmse_ab = np.sqrt(test_r_mse_ab)
            test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(euler_ab)))
            test_t_mse_ab = np.mean((translation_ab.detach().cpu().numpy() - translation_ab_pred_acc.detach().cpu().numpy()) ** 2)
            test_t_rmse_ab = np.sqrt(test_t_mse_ab)
            test_t_mae_ab = np.mean(np.abs(translation_ab.detach().cpu().numpy() - translation_ab_pred_acc.detach().cpu().numpy()))
            
            test_rotations_ba_pred_euler = npmat2euler(rotation_ba_pred_acc.detach().cpu().numpy(), 'xyz')
            test_r_mse_ba = np.mean((test_rotations_ba_pred_euler - np.degrees(euler_ba)) ** 2)
            test_r_rmse_ba = np.sqrt(test_r_mse_ba)
            test_r_mae_ba = np.mean(np.abs(test_rotations_ba_pred_euler - np.degrees(euler_ba)))
            test_t_mse_ba = np.mean((translation_ba.detach().cpu().numpy() - translation_ba_pred_acc.detach().cpu().numpy()) ** 2)
            test_t_rmse_ba = np.sqrt(test_t_mse_ba)
            test_t_mae_ba = np.mean(np.abs(translation_ba.detach().cpu().numpy() - translation_ba_pred_acc.detach().cpu().numpy()))

            
            output.write_file('%i,%s,%f,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f'
                        % (iteration,"A->B",test_loss,str(test_cycle_loss) if self._cfg.TRANSFORM.cycle else None,
                            test_mse_ab, test_rmse_ab, test_mae_ab, test_r_mse_ab, test_r_rmse_ab,
                            test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
            output.write_file('%i,%s,%f,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f'
                        % (iteration,"B->A",test_loss,str(test_cycle_loss) if self._cfg.TRANSFORM.cycle else None,
                            test_mse_ba, test_rmse_ba, test_mae_ba, test_r_mse_ba, test_r_rmse_ba,
                            test_r_mae_ba, test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))

            ret_loss.append(test_loss)
        output.close()
        xxx_file.close()
        return ret_loss

    def test_one_iteration(self, src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, rotation_ab_pred_acc, translation_ab_pred_acc, rotation_ba_pred_acc, translation_ba_pred_acc, src_acc, iteration):
        total_loss = 0
        total_cycle_loss = 0
        batch_size = src.shape[0]
       
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = self.estimate_rotation(src_acc, target)
        
        rotation_ab_pred_acc = torch.matmul(rotation_ab_pred_acc, rotation_ab_pred)
        translation_ab_pred_acc += translation_ab_pred
        rotation_ba_pred_acc = torch.matmul(rotation_ba_pred_acc, rotation_ba_pred)
        translation_ba_pred_acc += translation_ba_pred

        transformed_src = transform_point_cloud(src_acc, rotation_ab_pred, translation_ab_pred)
        transformed_target = transform_point_cloud(target, rotation_ba_pred_acc, translation_ba_pred_acc)
        # Save Transformeds
        for i in range(batch_size):
            output_folder = os.path.join(self._cfg.TRANSFORM.current_out_dir, str(i))
            os.makedirs(
                output_folder,
                exist_ok=True,
            )
            write_pcd_to_obj(os.path.join(output_folder, f"{iteration + 1}.obj"), transformed_src[i].squeeze().detach().cpu().numpy().T)

        # Losses
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1) 

        loss = F.mse_loss(torch.matmul(rotation_ab_pred_acc.transpose(2, 1), rotation_ab), identity) \
                + F.mse_loss(translation_ab_pred_acc, translation_ab)
        if self._cfg.TRANSFORM.cycle:
            rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred_acc, rotation_ab_pred_acc), identity.clone())
            translation_loss = torch.mean((torch.matmul(rotation_ba_pred_acc.transpose(2, 1),
                                                        translation_ab_pred_acc.view(batch_size, 3, 1)).view(batch_size, 3)
                                            + translation_ba_pred_acc) ** 2, dim=[0, 1])
            cycle_loss = rotation_loss + translation_loss

            loss = loss + cycle_loss * 0.1

        total_loss = loss.item() / self._cfg.TRANSFORM.test_batch_size

        if self._cfg.TRANSFORM.cycle:
            total_cycle_loss = cycle_loss.item() * 0.1

        mse_ab = torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item()
        mae_ab = torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item()

        mse_ba = torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item()
        mae_ba = torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item()
        return total_loss, total_cycle_loss, \
            mse_ab, mae_ab, \
            mse_ba, mae_ba, \
            rotation_ab_pred_acc, translation_ab_pred_acc, \
            rotation_ba_pred_acc, translation_ba_pred_acc, \
            transformed_src
    
    
    ########################### Training ###########################
    
    def train(self):
        self._model.train()
        self.train_iter = 0
        self.val_iter = 0
        self.early_stop_counter = 0
        self.best_test_loss = np.inf
        for epoch in range(self._cfg.TRANSFORM.epochs):        
            self.one_train_epoch(epoch)
            
                
    def one_train_epoch(self, epoch, mode="train", val_loss=0):
        self._model.train()
        with tqdm(self.train_data_loader, desc=mode, leave = False, ncols=150) as tepoch:
            tepoch.set_description(f"Epoch: {epoch + 1}/{self._cfg.TRANSFORM.epochs}")
            total_loss = 0
            for iteration, batch in enumerate(tepoch):
                
                self.optimizer.zero_grad()
                loss = self.general_step(batch, self.train_iter, "train")
                loss.backward()
                self.optimizer.step()
                tepoch.set_postfix_str(f"{mode}_loss = {loss}, val_loss = {val_loss}, Early-Stopping counter = {self.early_stop_counter}")
                self.scheduler.step()
                if (self.train_iter + 1) % self._cfg.TRANSFORM.val_every == 0:
                    val_loss = self.one_val_epoch(epoch)
                self.train_iter += 1
        self.tblog.add_scalar(f"Loss/Average_{mode}_loss", scalar_value=total_loss / (iteration + 1), global_step=self.train_iter, new_style=True)    
         
    
    def one_val_epoch(self, epoch):
        self._model.eval()
        with tqdm(self.test_data_loader, desc="Val", leave = False, ncols=150) as vepoch:
            vepoch.set_description(f"Valitdaion")
            total_loss = 0
            for iteration, batch in enumerate(vepoch):
                loss = self.general_step(batch, self.val_iter, "val").item()
                total_loss += loss
                vepoch.set_postfix_str(f"val_loss = {loss}")
                self.val_iter += 1
            
            total_loss /= (iteration + 1)
            if self.best_test_loss >= total_loss:
                self.early_stop_counter = 0
                self.best_test_loss = total_loss
            
                path = self._cfg.TRANSFORM.checkpoints_path / Path(f"model.best_epoch_{epoch + 1}_{self.val_iter}_valloss_{total_loss}.t7")
                os.makedirs(self._cfg.TRANSFORM.checkpoints_path, exist_ok=True)
                if torch.cuda.device_count() > 1:
                    torch.save(self._model.module.state_dict(), path)
                else:
                    torch.save(self._model.state_dict(), path)
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter == CFG.TRANSFORM.early_stop_thresh:
                    print(f"Early stopping with validation score {total_loss}")
                    exit()
            self._model.train()
            return total_loss
    
    
    def general_step(self, batch,  iteration, mode):
        x = batch[-1][PC_TYPES_LIST[PC_TYPES.PARTIAL_CLOUD]]
        gt = batch[-1][PC_TYPES_LIST[PC_TYPES.GT_CLOUD]]
        batch_size = x.shape[0]
        sources, targets, rotations_ab, translations_ab = [], [], [], []
        for i in range(len(x)):
            cloud = x[i][None, :]
            full_pc = gt[i][None, :]
            src, target, rotation_ab, translation_ab, _, _, _, _ = self.create_random_trans(cloud, full_pc)
            rotation_ab, translation_ab = rotation_ab.reshape(-1, *rotation_ab.shape), translation_ab.reshape(-1, *translation_ab.shape)
            sources.append(src);targets.append(target);rotations_ab.append(rotation_ab);translations_ab.append(translation_ab)
            
        src, target, rotation_ab, translation_ab = torch.cat(sources, axis=0), torch.cat(targets, axis=0), torch.cat(rotations_ab, axis=0), torch.cat(translations_ab, axis=0)

        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred= self._model(src, target)
            
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
            + F.mse_loss(translation_ab_pred, translation_ab)
        if self._cfg.TRANSFORM.cycle:
            rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
            translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
                                                        translation_ab_pred.view(batch_size, 3, 1)).view(batch_size, 3)
                                        + translation_ba_pred) ** 2, dim=[0, 1])
            cycle_loss = rotation_loss + translation_loss

            loss = loss + cycle_loss * 0.1

        self.tblog.add_scalar(f"Loss/{mode}",scalar_value=loss,  global_step=iteration, new_style=True)
        return loss