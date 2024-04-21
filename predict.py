import os, sys
from cog import BasePredictor, Input, Path
sys.path.append('/content/ControlNet-v1-1-nightly')
os.chdir('/content/ControlNet-v1-1-nightly')

from share import *
import config
from cldm.hack import hack_everything
hack_everything(clip_skip=2)

import cv2
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.lineart_anime import LineartAnimeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from PIL import Image

def process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, strength, scale, seed, eta, model, ddim_sampler, preprocessor):
    if det == 'Lineart_Anime':
        if not isinstance(preprocessor, LineartAnimeDetector):
            preprocessor = LineartAnimeDetector()
    with torch.no_grad():
        input_image = HWC3(input_image)

        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        control = 1.0 - torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)
        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        model.control_scales = [strength] * 13
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.preprocessor = None
        model_name = 'control_v11p_sd15s2_lineart_anime'
        self.model = create_model(f'./models/{model_name}.yaml').cpu()
        self.model.load_state_dict(load_state_dict('./models/anything-v3-full.safetensors', location='cuda'), strict=False)
        self.model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
    def predict(
        self,
        input_image: Path = Input(description="Input Image"),
        prompt: str = Input(''),
        det: str = Input(choices=['None','Lineart_Anime'], default='None'),
        a_prompt: str = Input('masterpiece, best quality, ultra-detailed, illustration, disheveled hair'),
        n_prompt: str = Input('longbody, lowres, bad anatomy, bad hands, missing fingers, pubic hair,extra digit, fewer digits, cropped, worst quality, low quality'),
        # num_samples: int = Input(default=1),
        image_resolution: int = Input(512),
        detect_resolution: int = Input(512),
        ddim_steps: int = Input(20),
        strength: float = Input(1.0),
        scale: float = Input(9.0),
        seed: int = Input(12345),
        eta: float = Input(1.0),
    ) -> Path:
        num_samples = 1
        image = cv2.imread(str(input_image))
        input_image = np.array(image)
        result = process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, strength, scale, seed, eta, self.model, self.ddim_sampler, self.preprocessor)
        image = cv2.cvtColor(result[1], cv2.COLOR_RGB2BGR)
        cv2.imwrite('/content/output.png', image)
        return Path('/content/output.png')