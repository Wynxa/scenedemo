# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Task-local dataset catalog for the construction predcls project."""

import copy
import os


class DatasetCatalog(object):
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DATA_DIR = os.path.join(REPO_ROOT, "datasets", "construction_vg_raw_train_test_full_20260703")

    DATASETS = {
        "ConstructionVG_predcls": {
            "img_dir": "vg/VG_100K",
            "roidb_file": "vg/VG-SGG-with-attri.h5",
            "dict_file": "vg/VG-SGG-dicts-with-attri.json",
            "image_file": "vg/image_data.json",
        },
    }

    @staticmethod
    def get(name, cfg):
        if "VG" in name:
            p = name.rfind("_")
            name, split = name[:p], name[p + 1:]
            assert name in DatasetCatalog.DATASETS and split in {"train", "val", "test"}
            data_dir = DatasetCatalog.DATA_DIR
            args = copy.deepcopy(DatasetCatalog.DATASETS[name])
            for k, v in args.items():
                args[k] = os.path.join(data_dir, v)
            args["split"] = split
            args["filter_non_overlap"] = (
                (not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX)
                and cfg.MODEL.RELATION_ON
                and cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP
            )
            args["filter_empty_rels"] = cfg.MODEL.RELATION_ON
            args["flip_aug"] = cfg.MODEL.FLIP_AUG
            args["custom_eval"] = cfg.TEST.CUSTUM_EVAL
            args["custom_path"] = cfg.TEST.CUSTUM_PATH
            args["num_im"] = cfg.DATASETS.debug_num_im
            args["num_val_im"] = cfg.DATASETS.debug_num_val_im
            return dict(factory="VGDataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = (
        "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/"
        "generalized_rcnn/model_final.pkl"
    )
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao",
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/") :]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        return "/".join([prefix, name])

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        name = name[len("Caffe2Detectron/COCO/") :]
        model_id, model_name = name.split("/")
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        return "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
