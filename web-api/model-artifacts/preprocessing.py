import cv2
import io
import numpy as np
import tempfile
import torch

from feature_extractor.module_clip import CLIP, convert_weights
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode


class TempModel(torch.nn.Module):
    def __init__(self, clip):
        super(TempModel, self).__init__()
        self.clip=clip

class Preprocessing:
    """
    This module is reposible to preprocess a video input from the requests into a visual representation
    """
    def __init__(self, feature_extractor_pt_path, args=None):
        self.feature_extractor = self._load_feature_extractor(feature_extractor_pt_path)
        self.feature_extractor.eval()
        self.transform = self._transform(224)

    def preprocess(self, requests, args):
        """
        Preprocess video input and return the visual representation along with the masking for transformer model
        """
        with torch.no_grad():
            video_mask = np.zeros((len(requests), args.max_frames), dtype=int)
            max_video_length = [0] * len(requests)
            video = np.zeros((len(requests), args.max_frames, 1, 3,
                                  224, 224), dtype=float)
            video = np.zeros((len(requests), args.max_frames, args.video_dim), dtype=float)
            for i,req in enumerate(requests):
                raw_video_data, shapes = self._transform_to_tensor(req)
                raw_video_data = raw_video_data['video']


                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    # L x T x 3 x H x W
                    raw_video_slice = self._process_raw_data(raw_video_data_clip)
                    if args.max_frames < raw_video_slice.shape[0]:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=args.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len

                    # extract clip
                    vid_tensor = torch.as_tensor(video_slice).float()
                    video_frame,num,channel,h,w = vid_tensor.shape
                    vid_tensor = vid_tensor.view(video_frame*num, channel, h, w)

                    video_frame,channel,h,w = vid_tensor.shape

                    extracted_feat = self.feature_extractor.encode_image(vid_tensor, video_frame=video_frame).float()
                    extracted_feat = extracted_feat.detach().numpy()

                    if slice_len < 1:
                        pass
                    else:
                        video[i][:slice_len] = extracted_feat

                else:
                    print("video error: {}".format(req))

            for i, v_length in enumerate(max_video_length):
                video_mask[i][:v_length] = [1] * v_length

        return torch.Tensor(video), torch.Tensor(video_mask)
        
    def _video_to_tensor(self, video, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.

        with tempfile.NamedTemporaryFile() as temp:
            temp.write(video)
            cap = cv2.VideoCapture(temp.name)

        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images, included = [], []

        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

        cap.release()

        if len(images) > 0:
            video_data = torch.tensor(np.stack(images))
        else:
            video_data = torch.zeros(1)
        return {'video': video_data},video_data.shape

    def _transform(self, n_px):
        return Compose([
                Resize(n_px, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

    def _transform_to_tensor(self, req):
        """
        transform to tensor.
        """
        # get video from the request
        videos = []
        video = req.get("data")
        if video is None:
            video = req.get("body")
        #create a stream from the encoded video
        video = io.BytesIO(video).read()

        image_input,shapes = self._video_to_tensor(video, self.transform, sample_fp=1)

        return image_input, shapes

    def _process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def _load_feature_extractor(self, feature_extractor_pt_path):
        clip_state_dict = torch.load(feature_extractor_pt_path, map_location='cpu')
        vit = "clip.visual.proj" in clip_state_dict
        assert vit
        vision_width = clip_state_dict["clip.visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in clip_state_dict.keys() if k.startswith("clip.visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = clip_state_dict["clip.visual.conv1.weight"].shape[-1]
        grid_size = round((clip_state_dict["clip.visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = clip_state_dict["clip.text_projection"].shape[1]
        context_length = clip_state_dict["clip.positional_embedding"].shape[0]
        vocab_size = clip_state_dict["clip.token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["clip.ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[3] for k in clip_state_dict if k.startswith(f"clip.transformer.resblocks")))


        print("\t embed_dim: {}".format(embed_dim))
        print("\t image_resolution: {}".format(image_resolution))
        print("\t vision_layers: {}".format(vision_layers))
        print("\t vision_width: {}".format(vision_width))
        print("\t vision_patch_size: {}".format(vision_patch_size))
        print("\t context_length: {}".format(context_length))
        print("\t vocab_size: {}".format(vocab_size))
        print("\t transformer_width: {}".format(transformer_width))
        print("\t transformer_heads: {}".format(transformer_heads))
        print("\t transformer_layers: {}".format(transformer_layers))

        cut_top_layer = 0
        linear_patch="2d"

        clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=linear_patch
        ).float()

        #cpu is not supported hal precision
        #convert_weights(clip)

        model = TempModel(clip)

        model = self._init_preweight(model, clip_state_dict)

        return model.clip


    def _init_preweight(self, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def _load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    _load(child, prefix + name + '.')

        _load(model, prefix='')

        if prefix is None and (task_config is None or task_config.local_rank == 0):
            # logger.info("-" * 20)
            print("-" * 20)
            if len(missing_keys) > 0:
                # logger.info("Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys))
                print("Weights of {} not initialized from pretrained model: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                # logger.info("Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
                print("Weights from pretrained model not used in {}: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            if len(error_msgs) > 0:
                # logger.error("Weights from pretrained model cause errors in {}: {}".format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))
                print("Weights from pretrained model cause errors in {}: {}"
                             .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))
        return model

