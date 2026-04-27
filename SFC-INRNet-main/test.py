import argparse
import os
import time
import threading
from collections import OrderedDict
import numpy as np
from skimage import measure
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import *
# Import your proposed model
from model.SFC_INRNet import SFC_INRNet


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    predict = (output > score_thresh).float()
    if len(target.shape) == 3:
        target = target.unsqueeze(dim=0)
        target = target.to('cuda', torch.float)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    # Intersection of prediction and Ground Truth
    intersection = predict * ((predict == target).float())

    tp = intersection.sum()  # True Positive
    fp = (predict * ((predict != target).float())).sum()  # False Positive (False Alarms)
    tn = ((1 - predict) * ((predict == target).float())).sum()  # True Negative
    fn = (((predict != target).float()) * (1 - predict)).sum()  # False Negative

    pos = tp + fn  # Total positive pixels in GT
    neg = fp + tn  # Total negative pixels in GT
    class_pos = tp + fp  # Total predicted positive pixels

    return tp, pos, fp, neg, class_pos


class SamplewiseSigmoidMetric(object):
    """Computes nIoU metric scores"""

    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.lock = threading.Lock()
        self.reset()

    def update(self, preds, labels):
        def evaluate_worker(self, label, pred):
            inter_arr, union_arr = batch_intersection_union_n(pred, label, self.nclass, self.score_thresh)
            with self.lock:
                self.total_inter = np.append(self.total_inter, inter_arr)
                self.total_union = np.append(self.total_union, union_arr)

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker, args=(self, label, pred))
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def get(self):
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        nIoU = IoU.mean()
        return nIoU

    def reset(self):
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])


def batch_intersection_union_n(output, target, nclass, score_thresh):
    """Calculates intersection and union for nIoU"""
    mini = 1
    maxi = 1
    nbins = 1
    outputnp = output.detach().cpu().numpy()
    predict = (outputnp > score_thresh).astype('int64')

    if len(target.shape) == 3:
        target = np.expand_dims(target.cpu().numpy(), axis=1).astype('int64')
    elif len(target.shape) == 4:
        target = target.cpu().numpy().astype('int64')
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * (predict == target)

    num_sample = intersection.shape[0]
    area_inter_arr = np.zeros(num_sample)
    area_pred_arr = np.zeros(num_sample)
    area_lab_arr = np.zeros(num_sample)
    area_union_arr = np.zeros(num_sample)

    for b in range(num_sample):
        area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
        area_inter_arr[b] = area_inter

        area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
        area_pred_arr[b] = area_pred

        area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
        area_lab_arr[b] = area_lab

        area_union = area_pred + area_lab - area_inter
        area_union_arr[b] = area_union

        assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    return area_inter_arr, area_union_arr


class ROCMetric05():
    """Computes ROC, Recall, Precision, and F-measure scores"""

    def __init__(self, nclass, bins):
        super(ROCMetric05, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            score_thresh = (0.0 + iBin) / self.bins
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)
        FP = self.fp_arr / (self.neg_arr + self.pos_arr)
        recall = self.tp_arr / (self.pos_arr + 0.001)
        precision = self.tp_arr / (self.class_pos + 0.001)
        # F-measure calculation (F1-score)
        f1_score = (2.0 * recall[5] * precision[5]) / (recall[5] + precision[5] + 0.00001)

        return tp_rates, fp_rates, recall, precision, FP, f1_score


class mIoU():
    """Computes pixAcc and IoU metric scores"""

    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def batch_pix_accuracy(output, target):
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


class PD_FA():
    """Computes Probability of Detection (Pd) and False Alarm Rate (Fa)"""

    def __init__(self, ):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64')

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)  # Total targets in Ground Truth
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        # Match predicted centroids with Ground Truth centroids
        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)
                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.dismatch_pixel += np.sum(self.dismatch)
        self.all_pixel += size[0] * size[1]
        self.PD += len(self.distance_match)

    def get(self):
        Final_FA = self.dismatch_pixel / self.all_pixel
        Final_PD = self.PD / self.target
        return Final_PD, float(Final_FA)


# ==============================================================================
# Testing Configuration
# ==============================================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch SFC-INRNet Test")
parser.add_argument('--ROC_thr', type=int, default=10, help='Number of thresholds for ROC curve')
parser.add_argument("--model_names", default=['SFC_INRNet'], type=list, help="Model name to evaluate")
parser.add_argument("--pth_dirs", default=['SFC_INRNet_best.pth.tar'], type=list)
parser.add_argument("--dataset_dir", default=r'./datasets', type=str, help="Dataset root directory")
parser.add_argument("--dataset_names", default=['NUDT-SIRST', 'NUAA-SIRST', 'IRSTD-1K'], type=list,
                    help="Datasets evaluated in the paper")
parser.add_argument("--img_norm_cfg", default=None, type=dict, help="Image normalization config")
parser.add_argument("--save_img", default=True, type=bool, help="Save the predicted mask images")
parser.add_argument("--save_img_dir", type=str, default=r'./results/', help="Path to save visualizations")
parser.add_argument("--save_log", type=str, default=r'./log/', help="Path of saved checkpoints")
parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")

global opt
opt = parser.parse_args()


def test():
    test_set = SFC_TestDataset(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    # Initialize Metrics
    IOU = mIoU()
    nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0)
    eval_05 = PD_FA()
    ROC_05 = ROCMetric05(nclass=1, bins=10)

    # Initialize Model (SFC-INRNet) and load weights
    net = SFC_INRNet().cuda()
    state_dict = torch.load(opt.pth_dir)

    new_state_dict = OrderedDict()
    for k, v in state_dict['state_dict'].items():
        name = k[6:] if k.startswith('module.') else k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()

    tbar = tqdm(test_loader, desc=f"Testing {opt.test_dataset_name}")
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, img_dir) in enumerate(tbar):
            img = img.cuda()
            gt_mask = gt_mask.cuda()

            # Forward Pass
            pred = net.forward(img)

            # Handle list/tuple outputs from Implicit Decoders / Deep Supervision
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]

            pred = pred[:, :, :size[0], :size[1]]
            gt_mask = gt_mask[:, :, :size[0], :size[1]]

            # Update Metrics
            IOU.update((pred > 0.5), gt_mask)
            nIoU_metric.update(pred, gt_mask)
            eval_05.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :].cpu(), size)
            ROC_05.update(pred, gt_mask)

            # Save Output Predictions
            if opt.save_img:
                img_save = transforms.ToPILImage()((pred[0, 0, :, :]).cpu())
                save_path = os.path.join(opt.save_img_dir, opt.test_dataset_name, opt.model_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                img_save.save(os.path.join(save_path, img_dir[0] + '.png'))

        # Fetch Final Results
        pixAcc, mIOU = IOU.get()
        nIoU = nIoU_metric.get()
        results2 = eval_05.get()  # Pd, Fa
        ture_positive_rate, false_positive_rate, recall, precision, FP, F_measure = ROC_05.get()

        # Log metrics strictly matching Table 1 of the manuscript
        print(f"\n[{opt.model_name} on {opt.test_dataset_name}]")
        print('pixAcc: %.4f | IoU: %.4f | nIoU: %.4f | Pd: %.4f | Fa: %.4f | F-measure: %.4f'
              % (pixAcc * 100, mIOU * 100, nIoU * 100, results2[0] * 100, results2[1] * 1e+6, F_measure * 100))

        # Write to log file
        opt.f.write('pixAcc: %.4f | IoU: %.4f | nIoU: %.4f | Pd: %.4f | Fa: %.4f | F-measure: %.4f\n'
                    % (pixAcc * 100, mIOU * 100, nIoU * 100, results2[0] * 100, results2[1] * 1e+6, F_measure * 100))


if __name__ == '__main__':
    if not os.path.exists(opt.save_log):
        os.makedirs(opt.save_log)

    log_name = 'test_results_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt'
    opt.f = open(os.path.join(opt.save_log, log_name), 'w')

    if not opt.pth_dirs:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            opt.f.write(opt.model_name + '_best.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                opt.pth_dir = os.path.join(opt.save_log, opt.dataset_name, opt.model_name + '_best.pth.tar')
                if os.path.exists(opt.pth_dir):
                    test()
                else:
                    print(f"Warning: Checkpoint not found at {opt.pth_dir}")
            opt.f.write('\n')
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    opt.test_dataset_name = dataset_name
                    opt.model_name = model_name
                    # Assuming pth_dir might just be the filename, we build the full path
                    full_pth = os.path.join(opt.save_log, dataset_name, pth_dir)
                    if not os.path.exists(full_pth):
                        continue
                    opt.train_dataset_name = dataset_name
                    opt.pth_dir = full_pth
                    test()
                    opt.f.write('\n')

    opt.f.close()
