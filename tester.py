from pathlib import Path
import os
import json
import utils

import torch

import visualize
import evaluate
from dataloader import remove_files_under_sample_size

class Tester:
    
    def __init__(self):
        return
    
    def _load_json(self, opt):
        
        keep_opt_list = ['phase', 'split_time']
        
        with open(os.path.join(opt.test_dir, "options.json"), 'r') as option:
            opt_dict = json.load(option)
        for k, v in opt_dict.items():
            if k not in keep_opt_list:
                setattr(opt, k, v)

        # Metric directory
        opt.result_image_dir = utils.create_folder_ifnotexist(Path(opt.log_dir) / "result_images")
        
        return opt
    
    def _load_model(self, opt, model):
        checkpoints = os.listdir(opt.checkpoint_dir)
        print(f"Possible loading models:{checkpoints}")
        checkpoint_file = checkpoints[-1]
        print(f"Load checkpoint file... {os.path.join(opt.checkpoint_dir, checkpoint_file)}")
        utils.load_checkpoint(model, os.path.join(opt.checkpoint_dir, checkpoint_file))
    
    def _set_properties(self, opt, model, loader_objs, device):
        self.opt = opt
        self.model = model.to(device)
        self.test_dataloader = loader_objs['test_dataloader']
        self.train_dataloader = loader_objs['train_dataloader']
        self.n_test_batches = loader_objs['n_test_batches']
        self.n_train_batches = loader_objs['n_train_batches']
        self.device = device
    
    @torch.no_grad()
    def infer_and_metrics(self):
        
        test_interp = True if not self.opt.extrap else False
        
        for it in range(self.n_test_batches):
            data_dict = utils.get_data_dict(self.test_dataloader)
            batch_dict = utils.get_next_batch(data_dict, test_interp=test_interp)

            preds, extra_info = self.model.get_reconstruction(time_steps_to_predict=batch_dict["tp_to_predict"],
                                                              truth=batch_dict["observed_data"],
                                                              truth_time_steps=batch_dict["observed_tp"],
                                                              mask=batch_dict["observed_mask"],
                                                              out_mask=batch_dict["mask_predicted_data"])

            b, _, c, h, w = batch_dict["data_to_predict"].size()
            selected_time_len = int(batch_dict["mask_predicted_data"][0].sum())
            batch_dict["data_to_predict"] = batch_dict["data_to_predict"][batch_dict["mask_predicted_data"].squeeze(-1).byte()].view(b, selected_time_len, c, h, w)

            visualize.save_test_images(opt=self.opt, preds=preds, batch_dict=batch_dict, path=self.opt.result_image_dir, index=it * self.opt.batch_size)

            if (it + 1) % 10 == 0:
                print(f"step: {it + 1:8d} testing...")
        
        pred_list = os.listdir(os.path.join(self.opt.result_image_dir, 'pred'))
        gt_list = os.listdir(os.path.join(self.opt.result_image_dir, 'gt'))
        
        evaluate.Evaluation(self.opt, pred_list, gt_list)