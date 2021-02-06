import matplotlib
matplotlib.use('Agg')

import torch
from torchvision.utils import save_image
import os

import utils


def save_test_images(opt, preds, batch_dict, path, index):
    preds = preds.cpu().detach()
    if opt.dataset == 'hurricane':
        gt = batch_dict['orignal_data_to_predict'].cpu().detach()
    else:
        gt = batch_dict['data_to_predict'].cpu().detach()

    b, t, c, h, w = gt.shape
    
    if opt.input_norm:
        preds = utils.denorm(preds)
        gt = utils.denorm(gt)
    
    os.makedirs(os.path.join(path, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(path, 'gt'), exist_ok=True)
    
    for i in range(b):
        for j in range(t):
            save_image(preds[i, j, ...], os.path.join(path, 'pred', f"pred_{index + i:03d}_{j:03d}.png"))
            save_image(gt[i, j, ...], os.path.join(path, 'gt', f"gt_{index + i:03d}_{j:03d}.png"))


def make_save_sequence(opt, batch_dict, res):
    """ 4 cases: (interp, extrap) | (regular, irregular) """
    
    b, t, c, h, w = batch_dict['observed_data'].size()

    # Filter out / Select by mask
    if opt.irregular:
        observed_mask = batch_dict["observed_mask"]
        mask_predicted_data = batch_dict["mask_predicted_data"]
        selected_timesteps = int(observed_mask[0].sum())
        
        
        if opt.dataset in ['hurricane']:
            batch_dict['observed_data'] = batch_dict['observed_data'][observed_mask.squeeze(-1).byte(), ...].view(b, selected_timesteps, c, h, w)
            batch_dict['data_to_predict'] = batch_dict['data_to_predict'][mask_predicted_data.squeeze(-1).byte(), ...].view(b, selected_timesteps, c, h, w)
        else:
            batch_dict['observed_data'] = batch_dict['observed_data'] * observed_mask.unsqueeze(-1).unsqueeze(-1)
            batch_dict['data_to_predict'] = batch_dict['data_to_predict'] * mask_predicted_data.unsqueeze(-1).unsqueeze(-1)
        
    # Make sequence to save
    pred = res['pred_y'].cpu().detach()
    
    if opt.extrap:
        inputs = batch_dict['observed_data'].cpu().detach()
        gt_to_predict = batch_dict['data_to_predict'].cpu().detach()
        gt = torch.cat([inputs, gt_to_predict], dim=1)
    else:
        gt = batch_dict['data_to_predict'].cpu().detach()

    time_steps = None

    if opt.input_norm:
        gt = utils.denorm(gt)
        pred = utils.denorm(pred)
    
    return gt, pred, time_steps


def save_extrap_images(opt, gt, pred, path, total_step):
    
    pred = pred.cpu().detach()
    gt = gt.cpu().detach()
    b, t, c, h, w = gt.shape
    
    # Padding zeros
    PAD = torch.zeros((b, t // 2, c, h, w))
    pred = torch.cat([PAD, pred], dim=1)
    
    save_me = []
    for i in range(min([b, 4])):  # save only 4 items
        row = torch.cat([gt[i], pred[i]], dim=0)
        if opt.input_norm:
            row = utils.denorm(row)
        if row.size(1) == 1:
            row = row.repeat(1, 3, 1, 1)
        save_me += [row]
    save_me = torch.cat(save_me, dim=0)
    save_image(save_me, os.path.join(path, f"image_{(total_step + 1):08d}.png"), nrow=t)


def save_interp_images(opt, gt, pred, path, total_step):
    
    pred = pred.cpu().detach()
    data = gt.cpu().detach()
    b, t, c, h, w = data.shape
    
    save_me = []
    for i in range(min([b, 4])):  # save only 4 items
        row = torch.cat([data[i], pred[i]], dim=0)
        if opt.input_norm:
            row = utils.denorm(row)
        if row.size(1) == 1:
            row = row.repeat(1, 3, 1, 1)
        save_me += [row]
    save_me = torch.cat(save_me, dim=0)
    save_image(save_me, os.path.join(path, f"image_{(total_step + 1):08d}.png"), nrow=t)


if __name__ == '__main__':
    pass
