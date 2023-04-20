# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/4/13 11:09
    @filename: metric.py
    @software: PyCharm
"""

import torch
import math

from .confusion_matrix import confusion_matrix

def lg10(x):
    return torch.div(torch.log(x), math.log(10))

def maxOfTwo(x, y):
    z = x.clone()
    maskYLarger = torch.lt(x, y)
    z[maskYLarger.detach()] = y[maskYLarger.detach()]
    return z

def nValid(x):
    return torch.sum(torch.eq(x, x).float())

def nNanElement(x):
    return torch.sum(torch.ne(x, x).float())

def getNanMask(x):
    return torch.ne(x, x)

def setNanToZero(input, target):
    nanMask = getNanMask(target)
    nValidElement = nValid(target)

    _input = input.clone()
    _target = target.clone()

    _input[nanMask] = 0
    _target[nanMask] = 0

    return _input, _target, nanMask, nValidElement

class Metric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        num_classes = 2 if self.num_classes == 1 else self.num_classes
        self.matrix = torch.zeros((num_classes, num_classes))

    def update(self, output, target):
        # if (output.dim() == 4 or target.dim() == 2) and self.num_classes != 1:
        #     output = torch.max(output, dim=1)[1]
        # if self.num_classes == 1:
        #     output = torch.where(output >= 0.5, 1, 0)
        # num_classes = 2 if self.num_classes == 1 else self.num_classes
        # matrix = confusion_matrix(output.detach().cpu(), target.detach().cpu(), num_classes)
        # # if self.matrix.device != matrix.device:
        # #     self.matrix = self.matrix.to(matrix.device)
        # self.matrix += matrix.detach().cpu()
        _output, _target, nanMask, self.nValidElement = setNanToZero(output, target)
        if (self.nValidElement.data.cpu().numpy() > 0):
            self.diffMatrix = torch.abs(_output - _target)

            self.realMatrix = torch.div(diffMatrix, _target)
            self.realMatrix[nanMask] = 0

            self.LG10Matrix = torch.abs(lg10(_output) - lg10(_target))
            self.LG10Matrix[nanMask] = 0
            yOverZ = torch.div(_output, _target)
            zOverY = torch.div(_target, _output)

            self.maxRatio = maxOfTwo(yOverZ, zOverY)

    def evaluate(self):
        result = dict()
        # FP = self.matrix.sum(0) - torch.diag(self.matrix)
        # FN = self.matrix.sum(1) - torch.diag(self.matrix)
        # TP = torch.diag(self.matrix)
        # TN = self.matrix.sum() - (FP + FN + TP)
        # precision = TP / (TP + FP)
        # acc = (TP + TN) / (TP+FP+FN+TN)
        # recall = TP / (TP + FN)
        # npv = TN/(TN+FN)
        # fnr = FN / (TP+FN)
        # fpr = FP / (FP+TN)
        # mcc = (TP*TN-FP*FN) / torch.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        # # f1 =  2 * (precision * recall) / (precision + recall)
        # specficity = TN / (TN + FP)
        # iou = TP / (TP + FN +FP)
        # dice = (2*TP) / (2*TP + FN + FP)
        # result["FP"] = FP
        # result["FN"] = FN
        # result["TP"] = TP
        # result["TN"] = TN
        # result["precision"] = precision
        # result["acc"] = acc
        # result["dice"] = dice
        # result["specifity"] = specficity
        # result["iou"] = iou
        # result["recall"] = recall
        # result["mk"] = precision + npv - 1
        # result["npv"] = npv
        # result["mcc"] = mcc
        # result["bm"] = (recall+specficity - 1)
        # result["fnr"] = fnr
        # result["fpr"] = fpr
        # result["tpr"] = recall
        # result["tnr"] = specficity
        if (self.nValidElement.data.cpu().numpy() > 0):
            result['MSE'] = torch.sum(torch.pow(self.diffMatrix, 2)) / self.nValidElement
            result['MAE'] = torch.sum(self.diffMatrix) / self.nValidElement
            result['ABS_REL'] = torch.sum(self.realMatrix) / self.nValidElement
            result['LG10'] = torch.sum(self.LG10Matrix) / self.nValidElement
            result['DELTA1'] = torch.sum(
            torch.le(self.maxRatio, 1.25).float()) / self.nValidElement
            result['DELTA2'] = torch.sum(
            torch.le(self.maxRatio, math.pow(1.25, 2)).float()) / self.nValidElement
            result['DELTA3'] = torch.sum(
            torch.le(self.maxRatio, math.pow(1.25, 3)).float()) / self.nValidElement
            result['MSE'] = float(result['MSE'].data.cpu().numpy())
            result['ABS_REL'] = float(result['ABS_REL'].data.cpu().numpy())
            result['LG10'] = float(result['LG10'].data.cpu().numpy())
            result['MAE'] = float(result['MAE'].data.cpu().numpy())
            result['DELTA1'] = float(result['DELTA1'].data.cpu().numpy())
            result['DELTA2'] = float(result['DELTA2'].data.cpu().numpy())
            result['DELTA3'] = float(result['DELTA3'].data.cpu().numpy())
        return result

class SeparateMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.ious = []
        self.dices = []

    def update(self, output, target):
        if (output.dim() == 4 or target.dim() == 2) and self.num_classes != 1:
            output = torch.max(output, dim=1)[1]
        if self.num_classes == 1:
            output = torch.where(output >= 0.5, 1, 0)
        num_classes = 2 if self.num_classes == 1 else self.num_classes
        matrix = confusion_matrix(output.detach().cpu(), target.detach().cpu(), num_classes)
        FP = matrix.sum(0) - torch.diag(matrix)
        FN = matrix.sum(1) - torch.diag(matrix)
        TP = torch.diag(matrix)
        TN = matrix.sum() - (FP + FN + TP)
        iou = TP / (TP + FN + FP)
        dice = (2 * TP) / (2 * TP + FN + FP)
        self.ious.append(iou)
        self.dices.append(dice)

    def evaluate(self):
        return self.dices, self.ious