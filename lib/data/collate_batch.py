# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from lib.scene_parser.rcnn.structures.image_list import to_image_list

class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0, produce=False):
        self.size_divisible = size_divisible
        self.produce = produce

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        if self.produce:
            img_idx = transposed_batch[1]
            img_ids = transposed_batch[2]
            return images, img_idx, img_ids
        else:
            targets = transposed_batch[1]
            img_ids = transposed_batch[2]
            img_name = transposed_batch[3]
            return images, targets, img_ids, img_name


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))
