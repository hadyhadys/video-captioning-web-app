import importlib
import inspect
import logging
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import zipfile

from pathlib import Path
from torch import nn
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import list_classes_from_module, load_label_mapping

logger = logging.getLogger(__name__)

class VideoCaptioning(BaseHandler):
    """
    Custom handler of TorchServe for video captioning.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = None
        self.preprocessing = None
        self.postprocessing = None
        self.inference = None
        self.args = self.Namespace(max_words=48, max_frames=20, video_dim=512, custom_input_dim=None,
                                    visual_num_hidden_layers=2, decoder_num_hidden_layers=2, stage_two=True)

    def handle(self, data, context):
        """
        Invoked by TorchServe for prediction request.
        Sequentially do preprocessing, inference, and postprocessing.
        """
        if not self.initialized:
            self.initialize(context)

        if data is None:
            return None

        video, video_mask = self.preprocess(data)
        preds = self.infer(video, video_mask)
        result = self.postprocess(preds)
        return result

    def initialize(self, context):
        """
        Invoked by TorchServe for initialization of model and required modules.
        """
        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")

        # Get the device, i.e., cuda or cpu
        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )

        # Extracting config files, including modules, feature extractor, and additional files for the model
        config_files = os.path.join(model_dir, "extra-files.zip")
        with zipfile.ZipFile(config_files, 'r') as zip_ref:
            for name in sorted(zip_ref.namelist(), reverse=True):
                try:
                    self._remove_path(os.path.join(model_dir,name))
                    zip_ref.extract(name)
                except BaseException as error:
                    print("An exception error: {}".format(error))
        
        # Initialize and load the model checkpoint
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)
        model_file = self.manifest["model"].get("modelFile", "")
        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_model(
                model_dir, model_file, model_pt_path)
            self.model.to(self.device)
        else:
            logger.debug("Loading torchscript model")
            self.model = torch.jit.load(model_pt_path)
        self.model.eval()

        logger.debug('Model file %s loaded successfully', model_pt_path)

        # Get the path to the checkpoint of the feature extractor
        feature_extractor_pt_dir = os.path.join(model_dir, "feature_extractor/checkpoint")
        feature_extractor_pt_file = [f for f in os.listdir(feature_extractor_pt_dir) if not f.startswith('.')][0]
        feature_extractor_pt_path = os.path.join(feature_extractor_pt_dir, feature_extractor_pt_file)

        # Load the modules of preprocessing, inference, and postprocessing
        self.preprocessing = self._load_preprocessing(feature_extractor_pt_path)
        self.inference = self._load_inference()
        self.postprocessing = self._load_postprocessing()

        self.initialized = True

    def preprocess(self, requests):
        """
        Preprocess the video input from the requests.
        """
        video, video_mask = self.preprocessing.preprocess(requests, self.args)
        return video, video_mask

    def infer(self, video, video_mask):
        """
        Given the data after preprocessing, perform inference using the model.
        We return the output, i.e. caption representation, of the model.
        """
        pred_list = self.inference.infer(self.model, video, video_mask, self.args)
        return pred_list

    def postprocess(self, pred_list):
        """
        Given the output from inference, postprocess the output to have a human-readable caption.
        """
        all_result_lists = self.postprocessing.postprocess(pred_list)
        return all_result_lists

    def _load_model(self, model_dir, model_file, model_pt_path):
        """
        Load the model for eager mode.
        """
        model_state_dict = torch.load(model_pt_path, map_location=self.map_location)

        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)

        model_class_definitions = sorted(model_class_definitions, key=self._sort_by_line_no)

        model_class = model_class_definitions[-1]

        model = model_class.from_pretrained('modules/bert-base-uncased', 'modules/visual-base', 'modules/decoder-base',
                                   cache_dir=None, state_dict=model_state_dict, task_config=self.args)

        return model

    def _load_preprocessing(self, feature_extractor_pt_path):
        """
        Load the preprocessing module.
        """
        module = importlib.import_module("preprocessing")
        model_class_definitions = list_classes_from_module(module)
        model_class_definitions = sorted(model_class_definitions, key=self._sort_by_line_no)
        model_class = model_class_definitions[-1](feature_extractor_pt_path)
        return model_class

    def _load_inference(self):
        """
        Load the inference module.
        """
        module = importlib.import_module("inference")
        model_class_definitions = list_classes_from_module(module)
        model_class_definitions = sorted(model_class_definitions, key=self._sort_by_line_no)
        model_class = model_class_definitions[-1]()
        return model_class

    def _load_postprocessing(self):
        """
        Load the posprocessing module.
        """
        module = importlib.import_module("postprocessing")
        model_class_definitions = list_classes_from_module(module)
        model_class_definitions = sorted(model_class_definitions, key=self._sort_by_line_no)
        model_class = model_class_definitions[-1]()
        return model_class


    def _sort_by_line_no(self, model_class_def):
        """
        Sort class definition by line no of the source code
        """
        source, line_no = inspect.findsource(model_class_def.__init__)
        return line_no


    def _remove_path(self, path):
        """
        Remove file or dir if exist.
        param <path> could either be relative or absolute.
        """
        if os.path.isfile(path) or os.path.islink(path):
            print('REMOVE FILE:',path)
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            print('REMOVE DIR:',path)
            os.rmdir(path)  # remove empty dir
        else:
            print("file {} is not a file or dir.".format(path))

    class Namespace:
        """
        Used to store the arguments needed for the model
        """
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
