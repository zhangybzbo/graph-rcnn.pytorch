import os
import datetime
import logging
import time
import numpy as np
import torch
import cv2
import json, h5py, pickle
from .data.build import build_data_loader
from .scene_parser.parser import build_scene_parser
from .scene_parser.parser import build_scene_parser_optimizer
from .scene_parser.rcnn.utils.metric_logger import MetricLogger
from .scene_parser.rcnn.utils.timer import Timer, get_time_str
from .scene_parser.rcnn.utils.comm import synchronize, all_gather, is_main_process, get_world_size
from .scene_parser.rcnn.utils.visualize import select_top_predictions, overlay_boxes, overlay_class_names
from .data.evaluation import evaluate, evaluate_sg
from .utils.box import bbox_overlaps

class SceneGraphGeneration:
    """
    Scene graph generation
    """
    def __init__(self, cfg, arguments, local_rank, distributed, produce):
        """
        initialize scene graph generation model
        """
        self.cfg = cfg
        self.arguments = arguments.copy()
        self.device = torch.device("cuda")

        # build data loader
        self.data_loader_train = build_data_loader(cfg, split="train", is_distributed=distributed)
        self.data_loader_test = build_data_loader(cfg, split="test", is_distributed=distributed, produce=produce)

        cfg.DATASET.IND_TO_OBJECT = self.data_loader_train.dataset.ind_to_classes
        cfg.DATASET.IND_TO_PREDICATE = self.data_loader_train.dataset.ind_to_predicates

        logger = logging.getLogger("scene_graph_generation.trainer")
        logger.info("Train data size: {}".format(len(self.data_loader_train.dataset)))
        logger.info("Test data size: {}".format(len(self.data_loader_test.dataset)))

        if not os.path.exists("freq_prior.npy"):
            logger.info("Computing frequency prior matrix...")
            fg_matrix, bg_matrix = self._get_freq_prior()
            prob_matrix = fg_matrix.astype(np.float32)
            prob_matrix[:,:,0] = bg_matrix

            prob_matrix[:,:,0] += 1
            prob_matrix /= np.sum(prob_matrix, 2)[:,:,None]
            # prob_matrix /= float(fg_matrix.max())
            np.save("freq_prior.npy", prob_matrix)

        # build scene graph generation model
        self.scene_parser = build_scene_parser(cfg); self.scene_parser.to(self.device)
        self.sp_optimizer, self.sp_scheduler, self.sp_checkpointer, self.extra_checkpoint_data = \
            build_scene_parser_optimizer(cfg, self.scene_parser, local_rank=local_rank, distributed=distributed)

        self.arguments.update(self.extra_checkpoint_data)

    def _get_freq_prior(self, must_overlap=False):

        fg_matrix = np.zeros((
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
            ), dtype=np.int64)

        bg_matrix = np.zeros((
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
        ), dtype=np.int64)

        for ex_ind in range(len(self.data_loader_train.dataset)):
            gt_classes = self.data_loader_train.dataset.gt_classes[ex_ind].copy()
            gt_relations = self.data_loader_train.dataset.relationships[ex_ind].copy()
            gt_boxes = self.data_loader_train.dataset.gt_boxes[ex_ind].copy()

            # For the foreground, we'll just look at everything
            o1o2 = gt_classes[gt_relations[:, :2]]
            for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
                fg_matrix[o1, o2, gtr] += 1

            # For the background, get all of the things that overlap.
            o1o2_total = gt_classes[np.array(
                self._box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
            for (o1, o2) in o1o2_total:
                bg_matrix[o1, o2] += 1

            if ex_ind % 20 == 0:
                print("processing {}/{}".format(ex_ind, len(self.data_loader_train.dataset)))

        return fg_matrix, bg_matrix

    def _box_filter(self, boxes, must_overlap=False):
        """ Only include boxes that overlap as possible relations.
        If no overlapping boxes, use all of them."""
        n_cands = boxes.shape[0]

        overlaps = bbox_overlaps(torch.from_numpy(boxes.astype(np.float)), torch.from_numpy(boxes.astype(np.float))).numpy() > 0
        np.fill_diagonal(overlaps, 0)

        all_possib = np.ones_like(overlaps, dtype=np.bool)
        np.fill_diagonal(all_possib, 0)

        if must_overlap:
            possible_boxes = np.column_stack(np.where(overlaps))

            if possible_boxes.size == 0:
                possible_boxes = np.column_stack(np.where(all_possib))
        else:
            possible_boxes = np.column_stack(np.where(all_possib))
        return possible_boxes

    def train(self):
        """
        main body for training scene graph generation model
        """
        start_iter = self.arguments["iteration"]
        logger = logging.getLogger("scene_graph_generation.trainer")
        logger.info("Start training")
        meters = MetricLogger(delimiter="  ")
        max_iter = len(self.data_loader_train)
        self.scene_parser.train()
        start_training_time = time.time()
        end = time.time()
        for i, data in enumerate(self.data_loader_train):
            data_time = time.time() - end
            self.arguments["iteration"] = i
            imgs, targets, _, _ = data
            imgs = imgs.to(self.device); targets = [target.to(self.device) for target in targets]
            loss_dict = self.scene_parser(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = loss_dict
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            self.sp_optimizer.zero_grad()
            losses.backward()
            self.sp_optimizer.step()
            self.sp_scheduler.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if i % 20 == 0 or i == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "model: {tag}",
                            "eta: {eta}",
                            "iter: {iter}/{max_iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        tag="scene_parser",
                        eta=eta_string,
                        iter=i, max_iter=max_iter,
                        meters=str(meters),
                        lr=self.sp_optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if (i + 1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                self.sp_checkpointer.save("checkpoint_{:07d}".format(i), **self.arguments)
            if (i + 1) == max_iter:
                self.sp_checkpointer.save("checkpoint_final", **self.arguments)

    def _accumulate_predictions_from_multiple_gpus(self, predictions_per_gpu):
        all_predictions = all_gather(predictions_per_gpu)
        if not is_main_process():
            return
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
        # convert a dict where the key is the index in a list
        image_ids = list(sorted(predictions.keys()))
        if len(image_ids) != image_ids[-1] + 1:
            logger = logging.getLogger("scene_graph_generation.inference")
            logger.warning(
                "Number of images that were gathered from multiple processes is not "
                "a contiguous set. Some images might be missing from the evaluation"
            )

        # convert to a list
        predictions = [predictions[i] for i in image_ids]
        return predictions

    def visualize_detection(self, dataset, img_ids, imgs, predictions):
        visualize_folder = "visualize"
        if not os.path.exists(visualize_folder):
            os.mkdir(visualize_folder)
        for i, prediction in enumerate(predictions):
            top_prediction = select_top_predictions(prediction)
            img = imgs.tensors[i].permute(1, 2, 0).contiguous().cpu().numpy() + np.array(self.cfg.INPUT.PIXEL_MEAN).reshape(1, 1, 3)
            result = img.copy()
            r = result[:,:,0].copy()
            result[:,:,0] = result[:,:,2]
            result[:,:,2] = r
            result = overlay_boxes(result, top_prediction)
            result = overlay_class_names(result, top_prediction, dataset.ind_to_classes)
            cv2.imwrite(os.path.join(visualize_folder, "detection_{}.jpg".format(img_ids[i].decode("utf-8"))), result)

    def test(self, timer=None, visualize=False):
        """
        main body for testing scene graph generation model
        """
        logger = logging.getLogger("scene_graph_generation.inference")
        logger.info("Start evaluating")
        self.scene_parser.eval()
        targets_dict = {}
        results_dict = {}
        if self.cfg.MODEL.RELATION_ON:
            results_pred_dict = {}
        cpu_device = torch.device("cpu")
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        reg_recalls = []
        for i, data in enumerate(self.data_loader_test, 0):
            imgs, targets, image_ids, img_name = data
            imgs = imgs.to(self.device); targets = [target.to(self.device) for target in targets]
            if i % 10 == 0:
                logger.info("inference on batch {}/{}...".format(i, len(self.data_loader_test)))
            with torch.no_grad():
                if timer:
                    timer.tic()
                output = self.scene_parser(imgs)
                if self.cfg.MODEL.RELATION_ON:
                    output, output_pred = output
                    output_pred = [o.to(cpu_device) for o in output_pred]
                ious = bbox_overlaps(targets[0].bbox, output[0].bbox)
                reg_recall = (ious.max(1)[0] > 0.5).sum().item() / ious.shape[0]
                reg_recalls.append(reg_recall)
                if timer:
                    torch.cuda.synchronize()
                    timer.toc()
                output = [o.to(cpu_device) for o in output]
                if visualize:
                    self.visualize_detection(self.data_loader_test.dataset, img_name, imgs, output)
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
            targets_dict.update(
                {img_id: target for img_id, target in zip(image_ids, targets)}
            )
            if self.cfg.MODEL.RELATION_ON:
                results_pred_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output_pred)}
                )
            if self.cfg.instance > 0 and i > self.cfg.instance:
                break
        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        num_devices = get_world_size()
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(self.data_loader_test.dataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(self.data_loader_test.dataset),
                num_devices,
            )
        )
        predictions = self._accumulate_predictions_from_multiple_gpus(results_dict)
        if self.cfg.MODEL.RELATION_ON:
            predictions_pred = self._accumulate_predictions_from_multiple_gpus(results_pred_dict)
        if not is_main_process():
            return

        output_folder = "results"
        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
            if self.cfg.MODEL.RELATION_ON:
                torch.save(predictions_pred, os.path.join(output_folder, "predictions_pred.pth"))

        extra_args = dict(
            box_only=False if self.cfg.MODEL.RETINANET_ON else self.cfg.MODEL.RPN_ONLY,
            iou_types=("bbox",),
            expected_results=self.cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=self.cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        )
        eval_det_results = evaluate(dataset=self.data_loader_test.dataset,
                        predictions=predictions,
                        output_folder=output_folder,
                        **extra_args)

        if self.cfg.MODEL.RELATION_ON:
            eval_sg_results = evaluate_sg(dataset=self.data_loader_test.dataset,
                            predictions=predictions,
                            predictions_pred=predictions_pred,
                            output_folder=output_folder,
                            **extra_args)
                            
    def list_detection(self, dataset, img_ids, predictions):
        ls = {}
        for i, prediction in enumerate(predictions):
            datum = {}
            # top_prediction = select_top_predictions(prediction)
            datum['scores'] = prediction.get_field("scores").cpu().numpy()
            datum['labels'] = prediction.get_field("labels").cpu().numpy()
            datum['names'] = [dataset.ind_to_classes[i] for i in datum['labels']]
            # datum['features'] = prediction.get_field("features").view(-1, 2048).cpu().numpy() 
            datum['boxes'] = prediction.bbox.cpu().numpy()
            ls[img_ids[i].decode("utf-8")] = datum
        return ls
        
    def list_rel(self, dataset, img_ids, objects, predictions):
        ls = {}
        for i, (object, prediction) in enumerate(zip(objects, predictions)):
            datum = []
            obj_score = object.get_field("scores").cpu().numpy()
            obj_label = object.get_field("labels").cpu().numpy()
            all_rels = prediction.get_field("idx_pairs").cpu().numpy()
            fp_pred = prediction.get_field("scores").cpu().numpy()
            for j in range(fp_pred.shape[0]):
                # datum : [(obj0), (obj1), rel]
                # obj0, obj1 : (idx_objpre, score, name)
                # rel : (score, name)
                obj0 = (all_rels[j, 0], obj_score[all_rels[j, 0]], dataset.ind_to_classes[obj_label[all_rels[j, 0]]])
                obj1 = (all_rels[j, 1], obj_score[all_rels[j, 1]], dataset.ind_to_classes[obj_label[all_rels[j, 1]]])
                pre_id = np.argmax(fp_pred[j, 1:]) + 1
                score = fp_pred[j, pre_id]
                assert score == fp_pred[j, 1:].max()
                rel = dataset.ind_to_predicates[pre_id]
                relation = (score, rel)
                if obj0[1] > 0.2 and obj1[1] > 0.2 and relation[0] > 0.2:
                    datum.append([obj0, obj1, relation])
            ls[img_ids[i].decode("utf-8")] = datum
        return ls

    def produce(self, timer=None, visualize=False):
        """
        main body for testing scene graph generation model
        """
        logger = logging.getLogger("scene_graph_generation.inference")
        logger.info("Start evaluating")
        self.scene_parser.eval()
        cpu_device = torch.device("cpu")
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        
        info = {}
        file_num = 0
        accu = 0
        save_x = np.zeros((10000, 100, 2048))
        save_b = np.zeros((10000, 100, 4))
        save_objs = {}
        save_rels = {}
        for i, data in enumerate(self.data_loader_test, 0):
            imgs, image_idx, image_ids = data
            imgs = imgs.to(self.device)
            if i % 10 == 0:
                logger.info("inference on batch {}/{}...".format(i, len(self.data_loader_test)))
            with torch.no_grad():
                if timer:
                    timer.tic()
                output = self.scene_parser(imgs)
                if self.cfg.MODEL.RELATION_ON:
                    output, output_pred = output
                    output_pred = [o.to(cpu_device) for o in output_pred]
                if timer:
                    torch.cuda.synchronize()
                    timer.toc()
                output = [o.to(cpu_device) for o in output]
                if visualize:
                    self.visualize_detection(self.data_loader_test.dataset, image_ids, imgs, output)
                save_objs.update(self.list_detection(self.data_loader_test.dataset, image_ids, output))
                
                for j, prediction in enumerate(output):
                    info[image_ids[j].decode("utf-8")] = {'file': file_num, 'idx': (accu % 10000), 'img': image_ids[j].decode("utf-8")}
                    num_obj = prediction.bbox.size(0)
                    save_x[accu % 10000, :num_obj, :] = prediction.get_field("features").view(-1, 2048).cpu().numpy() 
                    save_b[accu % 10000, :num_obj, :] = prediction.bbox.cpu().numpy()
                    accu += 1
                    assert (accu - 1) // 10000 == file_num
                    if accu % 10000 == 0:
                        with h5py.File("results/gqa_preobjects_%d.h5" % file_num, "w") as f:
                            dset1 = f.create_dataset("features", data=save_x, dtype='float32')
                            dset2 = f.create_dataset("bboxes", data=save_b, dtype='float32')
                            print('save', "results/gqa_preobjects_%d.h5" % file_num)
                            file_num += 1
                            save_x = np.zeros((10000, 100, 2048))
                            save_b = np.zeros((10000, 100, 4))
                
            if self.cfg.MODEL.RELATION_ON:
                save_rels.update(self.list_rel(self.data_loader_test.dataset, image_ids, output, output_pred))
            if self.cfg.instance > 0 and i > self.cfg.instance:
                break
                
        if accu % 10000 != 0:
            assert (accu) // 10000 == file_num
            with h5py.File("results/gqa_preobjects_%d.h5" % file_num, "w") as f:
                dset1 = f.create_dataset("features", data=save_x, dtype='float32')
                dset2 = f.create_dataset("bboxes", data=save_b, dtype='float32')
                print('save', "results/gqa_preobjects_%d.h5" % file_num)
        with open('results/gqa_preobjects_info.json', 'w') as f:
            json.dump(info, f)
            print('save', accu)
                
        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        num_devices = get_world_size()
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(self.data_loader_test.dataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(self.data_loader_test.dataset),
                num_devices,
            )
        )        
        if not is_main_process():
            return
        
        output_folder = "results"
        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            with open(os.path.join(output_folder, "predictions.pl"), 'wb') as f:
                pickle.dump(save_objs, f)
            if self.cfg.MODEL.RELATION_ON:
                with open(os.path.join(output_folder, "predictions_pred.pl"), 'wb') as f:
                    pickle.dump(save_rels, f)


def build_model(cfg, arguments, local_rank, distributed, produce=False):
    return SceneGraphGeneration(cfg, arguments, local_rank, distributed, produce)
