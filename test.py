import time,glob,os,cv2
import queue, threading

from typing import Optional, Tuple
import torch
from PIL import Image

import onnxruntime as onnxrt
import requests
from transformers import AutoConfig, AutoModelForVision2Seq, TrOCRProcessor, VisionEncoderDecoderModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from check_time import get_time
import numpy as np


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
device = torch.device("cpu")


#model_name = "microsoft/trocr-small-printed"
model_name = "raxtemur/trocr-base-ru"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)


start_time = time.time()


#optimum-cli export onnx -m microsoft/trocr-small-printed --task vision2seq-lm onnx/ --atol 1e-3

####
### import craft functions
##from craft_text_detector import (
##    read_image,
##    load_craftnet_model,
##    load_refinenet_model,
##    get_prediction,
##    export_detected_regions,
##    export_extra_results,
##    empty_cuda_cache
##)
##
##
##
### set image path and export folder directory
##image = 'price3.jpg' # can be filepath, PIL image or numpy array
##output_dir = 'outputs/'
##
### read image
##image = read_image(image)
##
### load models
##refine_net = load_refinenet_model(cuda=True)
##craft_net = load_craftnet_model(cuda=True)
##
### perform prediction
##prediction_result = get_prediction(
##    image=image,
##    craft_net=craft_net,
##    refine_net=refine_net,
##    text_threshold=0.7,
##    link_threshold=0.4,
##    low_text=0.4,
##    cuda=True,
##    long_size=1280
##)
##
### export detected text regions
##exported_file_paths = export_detected_regions(
##    image=image,
##    regions=prediction_result["boxes"],
##    output_dir=output_dir,
##    rectify=True
##)
##
### export heatmap, detection points, box visualization
##export_extra_results(
##    image=image,
##    regions=prediction_result["boxes"],
##    heatmaps=prediction_result["heatmaps"],
##    output_dir=output_dir
##)
##
### unload models from gpu
##empty_cuda_cache()
##
##quit()

files=[]
for file in glob.glob('outputs/image_crops/*.png'):        
        files.append(file)        

sorted_files = sorted(files, key=lambda x: int(x.split('_')[2].split('.')[0]))
##
###29.7 sec

##
##
###очереди https://docs-python.ru/standart-library/modul-queue-python/obrabotka-ocheredi-neskolko-potokov/
##
##
### данные для очереди
#data = list(range(10,150,15))
data=sorted_files
#print('--> Start list\n', data)


class ORTEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main_input_name = "pixel_values"
        self._device = device
        self.session = onnxrt.InferenceSession(
            "onnx2/encoder_model.onnx", providers=["CPUExecutionProvider"]
        )
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ) -> BaseModelOutput:

        onnx_inputs = {"pixel_values": pixel_values.cpu().detach().numpy()}

        # Run inference
        outputs = self.session.run(None, onnx_inputs)
        last_hidden_state = torch.from_numpy(outputs[self.output_names["last_hidden_state"]]).to(self._device)

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._device = device

        self.session = onnxrt.InferenceSession(
            "onnx2/decoder_model.onnx", providers=["CPUExecutionProvider"]
        )

        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
    ) -> Seq2SeqLMOutput:

        onnx_inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
        }

        if "attention_mask" in self.input_names:
            onnx_inputs["attention_mask"] = attention_mask.cpu().detach().numpy()

        # Add the encoder_hidden_states inputs when needed
        if "encoder_hidden_states" in self.input_names:
            onnx_inputs["encoder_hidden_states"] = encoder_hidden_states.cpu().detach().numpy()

        # Run inference
        outputs = self.session.run(None, onnx_inputs)

        logits = torch.from_numpy(outputs[self.output_names["logits"]]).to(self._device)
        return Seq2SeqLMOutput(logits=logits)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_hidden_states=None, **kwargs):
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
        }


class ORTModelForVision2Seq(VisionEncoderDecoderModel, GenerationMixin):
    def __init__(self, *args, **kwargs):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        self._device = device

        self.encoder = ORTEncoder()
        self.decoder = ORTDecoder()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(pixel_values=pixel_values.to(device))

        # Decode
        decoder_attention_mask = decoder_input_ids.new_ones(decoder_input_ids.shape)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
        )

        return Seq2SeqLMOutput(
            logits=decoder_outputs.logits,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs):

        return {
            "decoder_input_ids": input_ids,
            "decoder_atttention_mask": input_ids,
            "encoder_outputs": encoder_outputs,
        }

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, value: torch.device):
        self._device = value

    def to(self, device):
        self.device = device
        return self

processor = TrOCRProcessor.from_pretrained(model_name)
model = ORTModelForVision2Seq()
model = model.to(device)


def test_ort(pixel_values):
##    processor = TrOCRProcessor.from_pretrained(model_name)
##    model = ORTModelForVision2Seq()
##    model = model.to(device)

    start = time.time()

    model.config.decoder_start_token_id = 2
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.pad_token_id = model.config.decoder.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = model.config.decoder.eos_token_id = processor.tokenizer.sep_token_id

    generated_ids = model.generate(pixel_values.to(device))

    end = time.time()
    model_output = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, device=device)[0]
    print("ORT time: ", end - start, model_output)


def test_original(img):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    start = time.time()
    generated_ids = model.generate(pixel_values.to(device))
    end = time.time()

    model_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("Original time: ", end - start, model_output)

def recognition_image(image):
        #crop_img = cv2.imread(image)
        #i = cv2.resize(crop_img, (int(crop_img.shape[1]*1.3), int(crop_img.shape[0]*1.3)),interpolation = cv2.INTER_LINEAR)
        #test_original()
        test_ort(image)

for i in data:
        crop_img = cv2.imread(i)   
        i = cv2.resize(crop_img, (int(crop_img.shape[1]*1.3), int(crop_img.shape[0]*1.3)),interpolation = cv2.INTER_LINEAR)
        price_crops = np.array(i)

        pixel_values = processor(price_crops, return_tensors="pt").pixel_values
        #pixel_values = pixel_values.to(device)
        #test_original(pixel_values)
        test_original(pixel_values)
        test_ort(pixel_values)
