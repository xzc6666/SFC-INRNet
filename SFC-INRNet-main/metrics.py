import numpy as np
import torch
from skimage import measure


# ==============================================================================
# SFC-INRNet Evaluation Metrics
# Specifically designed to calculate IoU, Probability of Detection (Pd),
# False-alarm Rate (Fa), and ROC/F-measure for Infrared Small Target Detection.
# ==============================================================================

def cal_tp_pos_fp_neg(output, target, score_thresh):
    """
    Calculate True Positive, Total Positive, False Positive, Total Negative,
    and Predicted Positive pixels based on a given threshold.
    """
    predict = (output > score_thresh).float()

    if len(target.shape) == 3:
        target = target.unsqueeze(dim=0)
        target = target.to(output.device, dtype=torch.float)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension.")

    # Intersection of prediction and Ground Truth
    intersection = predict * ((predict == target).float())

    tp = intersection.sum()  # True Positives
    fp = (predict * ((predict != target).float())).sum()  # False Positives (False Alarms)
    tn = ((1 - predict) * ((predict == target).float())).sum()  # True Negatives
    fn = (((predict != target).float()) * (1 - predict)).sum()  # False Negatives

    pos = tp + fn  # Total actual positive pixels
    neg = fp + tn  # Total actual negative pixels
    class_pos = tp + fp  # Total predicted positive pixels

    return tp, pos, fp, neg, class_pos


class SFC_ROC():
    """
    Computes pixel-level accuracy, ROC curve components (TPR, FPR),
    Precision, Recall, and F-measure.
    """

    def __init__(self, bins=10):
        super(SFC_ROC, self).__init__()
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)

    def update(self, preds, labels):
        # Iterate over uniformly distributed thresholds in [0, 1]
        for iBin in range(self.bins + 1):
            score_thresh = (0.0 + iBin) / self.bins
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds, labels, score_thresh)

            self.tp_arr[iBin] += i_tp.item() if torch.is_tensor(i_tp) else i_tp
            self.pos_arr[iBin] += i_pos.item() if torch.is_tensor(i_pos) else i_pos
            self.fp_arr[iBin] += i_fp.item() if torch.is_tensor(i_fp) else i_fp
            self.neg_arr[iBin] += i_neg.item() if torch.is_tensor(i_neg) else i_neg
            self.class_pos[iBin] += i_class_pos.item() if torch.is_tensor(i_class_pos) else i_class_pos

    def get(self):
        epsilon = 1e-5
        # Calculate rates
        tp_rates = self.tp_arr / (self.pos_arr + epsilon)  # Recall (TPR)
        fp_rates = self.fp_arr / (self.neg_arr + epsilon)  # FPR
        FP = self.fp_arr / (self.neg_arr + self.pos_arr + epsilon)

        recall = self.tp_arr / (self.pos_arr + epsilon)
        precision = self.tp_arr / (self.class_pos + epsilon)

        # Calculate F-measure (F1-score) using the middle threshold (idx 5 for bins=10)
        idx = self.bins // 2
        f_measure = (2.0 * recall[idx] * precision[idx]) / (recall[idx] + precision[idx] + epsilon)

        return tp_rates, fp_rates, recall, precision, FP, f_measure

    def reset(self):
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)


class SFC_mIoU():
    """
    Computes pixel Accuracy and Intersection over Union (IoU) metric scores.
    """

    def __init__(self):
        super(SFC_mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        epsilon = np.spacing(1)
        pixAcc = 1.0 * self.total_correct / (epsilon + self.total_label)
        IoU = 1.0 * self.total_inter / (epsilon + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


class SFC_PD_FA():
    """
    Computes target-level Probability of Detection (Pd) and pixel-level False-alarm Rate (Fa).
    Targets are matched based on centroid distance threshold.
    """

    def __init__(self):
        super(SFC_PD_FA, self).__init__()
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits = np.array(preds.cpu()).astype('int64')
        labelss = np.array(labels.cpu()).astype('int64')

        # Extract connected components using skimage.measure
        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)  # Total number of targets in Ground Truth

        image_area_total = []
        image_area_match = []
        distance_match = []

        for K in range(len(coord_image)):
            image_area_total.append(np.array(coord_image[K].area))

        # Centroid matching for Pd
        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)

                # Match threshold (e.g., 3 pixels)
                if distance < 3:
                    distance_match.append(distance)
                    image_area_match.append(np.array(coord_image[m].area))
                    del coord_image[m]  # Remove matched prediction
                    break

        # False alarms computation
        dismatch = [x for x in image_area_total if x not in image_area_match]
        self.dismatch_pixel += np.sum(dismatch)

        self.all_pixel += size[0] * size[1]
        self.PD += len(distance_match)

    def get(self):
        # Prevent division by zero
        target_cnt = self.target if self.target > 0 else 1
        pixel_cnt = self.all_pixel if self.all_pixel > 0 else 1

        Final_FA = self.dismatch_pixel / pixel_cnt
        Final_PD = self.PD / target_cnt
        return Final_PD, float(Final_FA)

    def reset(self):
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0


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
