import einops
import numpy as np
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.generation import GenerationConfig, GenerationMixin
from transformers import AutoConfig
import torch
import os
import openvino as ov

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from typing import Optional, Tuple, List
from openvino.runtime import opset13
import logging as log
from pathlib import Path
import numpy as np
import nncf

def model_has_state(ov_model: ov.Model):
    # TODO: Provide a better way based on the variables availability, but OV Python API doesn't expose required methods
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):   # TODO: Can we derive the dimensions from the model topology?
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    gather_dim: int,
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`List[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`List[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`List[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`List[str]`):
            list of names for key value input layers
        key_value_output_names (`List[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model):
    key_value_input_names = [
        key.get_any_name() for key in ov_model.inputs if any("key_values" in key_name for key_name in key.get_names())
    ]
    key_value_output_names = [
        key.get_any_name() for key in ov_model.outputs if any("present" in key_name for key_name in key.get_names())
    ]
    not_kv_inputs = [
        input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())
    ]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1
    
    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
    )


class BaseModel():
    def __init__(
        self,
        device='CPU',
        fp16=False,
    ):
        self.device = device
        self.name = "XTTS-V2 Model"


    def get_model(self):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

class GPTInferStatefulModel(BaseModel):
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
        int4_compress=False,
    ):
        self.name = "GPT Model"
        self.model = model
        self.device = device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.int4_compress = int4_compress
        self.inputs_dict = {}

    def get_model(self):
        return self.model.gpt.gpt_inference
    
    def get_input_names(self):
        inputs = []
        for idx in range(30):
            inputs.append(f"past_key_values.{idx}.key")
            inputs.append(f"past_key_values.{idx}.value")
        inputs.extend(['attention_mask', 'position_ids', 'inputs_embeds'])
        return inputs

    def get_output_names(self):
        outputs = ['lm_logits']
        for idx in range(30):
            outputs.append(f"present.{idx}.key")
            outputs.append(f"present.{idx}.value")
        return outputs
    
    def save_tokenizer(self, tokenizer, out_dir):
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception as e:
            log.error(f'tokenizer loading failed with {e}')
    
    def convert_sdpa_ov(self):
        language_model = self.get_model()
        input_ids = torch.randint(
                1, 255, ( 1, 99, 1024 )
            )
        # 'attention_mask': ['batch', 't']
        attention_mask = torch.from_numpy(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1]]))
                    # "position_ids": ['batch', 't']
        position_ids = torch.from_numpy(np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
         54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
         72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
         90, 91, 92, 93, 94, 95, 96, 97, 98]]))
        
        pkv = language_model(inputs_embeds=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=True, return_dict=False)[1]

        language_model.config.torchscript = True
        ov_model = ov.convert_model(
            language_model,
            example_input={
                "inputs_embeds": torch.rand(( 1, 1, 1024 ), dtype=torch.float32),
                "attention_mask": torch.from_numpy(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1]], dtype=np.float32)),
                "position_ids": torch.from_numpy(np.array([[99]], dtype=np.float32)),
                "past_key_values": pkv,
             },
        )

        print("llm stateful model input: ", ov_model.inputs)

        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        patch_stateful(ov_model)
        ov.save_model(ov_model, Path(f"{self.ov_model_path}/gpt_infer_stateful.xml"))
        # self.save_tokenizer(self.tokenizer, self.ov_model_path)
        # self.model.config.save_pretrained(self.ov_model_path)

        if self.int4_compress:
            compression_configuration = {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 128,
                "ratio": 1,
            }
            ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
            ov.save_model(ov_compressed_model, Path(f"{self.ov_model_path}/gpt_infer_stateful_int4.xml"))


class GPTModel(BaseModel):
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
    ):
        self.name = "GPT Model"
        self.model = model
        self.device=device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.inputs_dict = {}

    def get_model(self):
        return self.model.gpt

    def get_input_names(self):
        return ['text_input', 'text_lengths', 'audio_codes', 'wav_lengths', 
                #'cond_mels', 'cond_idxs', 'cond_lens', 
                'cond_latents', 
                #'return_attentions', 'return_latent',
                ]

    def get_output_names(self):
        return ['mel_logits']

    def get_dynamic_axes(self):
        return {
            'text_input': {0:'batch', 1:'t'},
            'text_lengths': {0:'batch'},
            'audio_codes': {0:'batch', 1:'t'},
            'wav_lengths': {0:'length'},
            'cond_latents': {0:'batch'},
                }

    def get_sample_input(self):
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            # 'text_input': ['batch', 't']
            torch.ones(
                1,
                66,
                dtype=torch.int32,
            ),
            # 'text_lengths': ['batch']
            torch.ones(
                1,
                dtype=torch.int32,
            ),
            # "audio_codes": ['batch', 't']
            torch.ones(
                1,
                243,
                dtype=torch.int32,
            ),
            # 'wav_lengths': ['batch']
            torch.ones(
                1,
                dtype=torch.int32,
            )*121856,
            # 'cond_mels', 'cond_idxs', 'cond_lens',
            None, None, None,
            #cond_latents [b , 32, 1024]
            torch.rand(
                (1, 32, 1024),
                dtype=torch.float32,
            ),
            # return_attentions, return_latent
            False, True
        )
    def convert_sdpa_ov(self):
        tmp_onnx_path = '/tmp/tmp_gpt.onnx'
        torch.onnx.export(self.get_model(),
            self.get_sample_input(),
            tmp_onnx_path,
            input_names=self.get_input_names(),
            output_names=self.get_output_names(),
            dynamic_axes=self.get_dynamic_axes()
        )

        self.gpt2_ov_model = ov.convert_model(tmp_onnx_path)
        # Check if the file exists
        if os.path.exists(tmp_onnx_path):
            # Delete the file
            os.remove(tmp_onnx_path)
        ov.save_model(self.gpt2_ov_model, Path(f"{self.ov_model_path}/gpt.xml"))

    # def convert_sdpa_ov(self):
    #     gpt_model = self.get_model()        
    #     ov_model = ov.convert_model(
    #         gpt_model,
    #         example_input={
    #             'text_inputs': torch.ones(1, 66, dtype=torch.int32), 
    #             'text_lengths': torch.ones(1, dtype=torch.int32), 
    #             'audio_codes': torch.ones(1, 243, dtype=torch.int32), 
    #             'wav_lengths': torch.ones(1, dtype=torch.int32)*121856,
    #             'cond_latents': torch.rand((1, 32, 1024), dtype=torch.float32), 
    #          },
    #     )
    #     for input, input_name in zip(ov_model.inputs, self.get_input_names()):
    #         input.get_tensor().set_names({input_name})
    #     for output, output_name in zip(ov_model.outputs, self.get_output_names()):
    #         output.get_tensor().set_names({output_name})
        
    #     ov.save_model(ov_model, Path(f"{self.ov_model_path}/gpt.xml"))

class HifiModel(BaseModel):
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
    ):
        self.name = "Hifi Model"
        self.model = model
        self.device=device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.inputs_dict = {}
        self.batch_size = 1

    def get_model(self):
        return self.model.hifigan_decoder

    def get_input_names(self):
        return ['latents', 'speaker_embedding']

    def get_output_names(self):
        return ['wave_data']

    def get_dynamic_axes(self):
        return {
            'latents': {0:'batch', 1:'t'},
                }

    def get_sample_input(self):
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            # 'latents': ['batch', 't', 1024]
            torch.rand(
                ( 1, 113, 1024 ),
                dtype=torch.float32,
            ),
            # 'speaker_embedding': ['1, 512, 1']
            torch.rand(
                (1, 512, 1),
                dtype=torch.float32,
            ),
        )
    
    def convert_sdpa_ov(self):
        tmp_onnx_path = '/tmp/tmp_hifi.onnx'
        torch.onnx.export(self.get_model(),
            self.get_sample_input(),
            tmp_onnx_path,
            input_names=self.get_input_names(),
            output_names=self.get_output_names(),
            dynamic_axes=self.get_dynamic_axes()
        )

        self.hifi_ov_model = ov.convert_model(tmp_onnx_path, input={"latents": ([1, -1, 1024], ov.Type.f32), "speaker_embedding": ([1, 512, 1], ov.Type.f32)})
        
        # Check if the file exists
        if os.path.exists(tmp_onnx_path):
            # Delete the file
            os.remove(tmp_onnx_path)

        ov.save_model(self.hifi_ov_model, Path(f"{self.ov_model_path}/hifi.xml"))

    # def convert_sdpa_ov(self):
    #     hifi_model = self.get_model()        
    #     ov_model = ov.convert_model(
    #         hifi_model,
    #         example_input={
    #             'latents': torch.rand(( 1, 113, 1024 ),dtype=torch.float32), 
    #             'speaker_embedding': torch.rand(( 1, 512, 1 ),dtype=torch.float32), 
    #          },
    #     )
    #     for input, input_name in zip(ov_model.inputs, self.get_input_names()):
    #         input.get_tensor().set_names({input_name})
    #     for output, output_name in zip(ov_model.outputs, self.get_output_names()):
    #         output.get_tensor().set_names({output_name})
        
    #     ov.save_model(ov_model, Path(f"{self.ov_model_path}/hifi.xml"))

class GPTInferPastModel(BaseModel):
    def __init__(
        self,
        model=None,
        device='CPU',
        ov_model_path = None,
        int4_compress=False,
        fp16=False,
    ):
        self.name = "GPT Model"
        self.model = model
        self.fp16=fp16
        self.device = device
        self.ov_model_path = ov_model_path
        self.int4_compress = int4_compress
        self.inputs_dict = {}

    def get_model(self):
        return self.model.gpt.gpt_inference

    def get_input_names(self):
        # inputs = ['inputs_embeds']
        inputs = []
        for idx in range(30):
            inputs.append(f"past_key_values.{idx}.key")
            inputs.append(f"past_key_values.{idx}.value")
        inputs.append('attention_mask')
        inputs.append('position_ids')
        inputs.append('inputs_embeds')
        return inputs

    def get_output_names(self):
        outputs = ['lm_logits']
        for idx in range(30):
            outputs.append(f"present.{idx}.key")
            outputs.append(f"present.{idx}.value")
        return outputs

    def convert_past_sdpa_ov(self):
        language_model = self.get_model()
        inputs_embeds = torch.randint(
                1, 255, ( 1, 99, 1024 )
            )
        # 'attention_mask': ['batch', 't']
        attention_mask = torch.from_numpy(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1]]))
        # "position_ids": ['batch', 't']
        position_ids = torch.from_numpy(np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
         54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
         72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
         90, 91, 92, 93, 94, 95, 96, 97, 98]]))
        
        pkv = language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, use_cache=True, return_dict=False)[1]

        language_model.config.torchscript = True
        ov_model = ov.convert_model(
            language_model,
            example_input={
                "inputs_embeds": torch.rand(( 1, 1, 1024 ), dtype=torch.float32),
                "attention_mask": torch.from_numpy(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1]], dtype=np.float32)),
                "position_ids": torch.from_numpy(np.array([[99]], dtype=np.float32)),
                "past_key_values": pkv,
             },
        )
        
        # print("model inputs: ", ov_model.inputs)
        # print("custom model inputs: ", self.get_input_names())
        # assert ov_model.inputs[0].get_names().pop() == 'input_ids'
        # assert ov_model.inputs[-1].get_names().pop() == 'position_ids'
        # assert ov_model.inputs[-2].get_names().pop() == 'attention_mask'

        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        
        ov.save_model(ov_model, Path(f"{self.ov_model_path}/gpt_infer_past.xml"))
        # self.save_tokenizer(self.tokenizer, self.ov_model_path)
        # self.model.config.save_pretrained(self.ov_model_path)

        if self.int4_compress:
            compression_configuration = {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 128,
                "ratio": 1,
            }
            ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
            ov.save_model(ov_compressed_model, Path(f"{self.ov_model_path}/gpt_infer_past_int4.xml"))
    
class GPTInferModel(BaseModel):
    def __init__(
        self,
        model=None,
        device='CPU',
        ov_model_path=None,
        int4_compress=False,
        fp16=False,
    ):
        self.name = "GPT Model"
        self.model = model
        self.device=device
        self.ov_model_path = ov_model_path
        self.int4_compress = int4_compress
        self.fp16=fp16
        self.inputs_dict = {}

    def get_model(self):
        return self.model.gpt.gpt_inference

    def get_input_names(self):
        return [ 'attention_mask', 'position_ids', 'inputs_embeds',
                ]

    def get_output_names(self):
        outputs = ['lm_logits']
        for idx in range(30):
            outputs.append(f"present.{idx}.key")
            outputs.append(f"present.{idx}.value")
        return outputs

    def convert_sdpa_ov(self):
        language_model = self.get_model()
        # 'attention_mask': ['batch', 't']
        attention_mask = torch.from_numpy(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1]], dtype=np.float32))
        # "position_ids": ['batch', 't']
        position_ids = torch.from_numpy(np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
         54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
         72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
         90, 91, 92, 93, 94, 95, 96, 97, 98]], dtype=np.float32))
        
        language_model.config.torchscript = True
        ov_model = ov.convert_model(
            language_model,
            example_input={
                "inputs_embeds": torch.rand(( 1, 99, 1024 ), dtype=torch.float32),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
             },
        )

        print('gpt_infer_model: ', ov_model.inputs)

        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/gpt_infer.xml"))
        # self.save_tokenizer(self.tokenizer, self.ov_model_path)
        # self.model.config.save_pretrained(self.ov_model_path)

        if self.int4_compress:
            compression_configuration = {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 128,
                "ratio": 1,
            }
            ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
            ov.save_model(ov_compressed_model, Path(f"{self.ov_model_path}/gpt_infer_int4.xml"))

class Xttsv2_OV():
    def __init__(self, model=None, ov_model_path='/tmp/xttsv2_ov/', device='CPU', int4_compress=False):

        self.model = model

        self.int4_compress = int4_compress
        self.core = ov.Core()
        self.gpt_model = GPTModel(model=self.model, ov_model_path=ov_model_path, device=device)
        self.hifi_model = HifiModel(model=self.model, ov_model_path=ov_model_path, device=device)

        # self.gpt_infer_stateful_model = GPTInferStatefulModel(model=self.model, ov_model_path=ov_model_path, device=device, int4_compress=self.int4_compress)
        self.gpt_infer_model = GPTInferModel(model=self.model, ov_model_path=ov_model_path, device=device, int4_compress=self.int4_compress)
        self.gpt_infer_past_model = GPTInferPastModel(model=self.model, ov_model_path=ov_model_path, device=device, int4_compress=self.int4_compress)

    def export_xttsv2_to_ov(self):
        # self.gpt_infer_stateful_model.convert_sdpa_ov()
        self.gpt_infer_model.convert_sdpa_ov()
        self.gpt_infer_past_model.convert_past_sdpa_ov()
        self.hifi_model.convert_sdpa_ov()
        self.gpt_model.convert_sdpa_ov()    

class OVXttsForCausalLM():
    def __init__(
        self,
        core=None,
        ov_model_path=None,
        device='CPU',
        int4_compress=False
    ):
        self.ov_model_path = ov_model_path
        self.core = core
        self.ov_device = device
        self.int4_compress = int4_compress

        if int4_compress:
            self.llm_past_model = core.read_model(Path(f"{ov_model_path}/gpt_infer_past_int4.xml"))
        else:
            self.llm_past_model = core.read_model(Path(f"{ov_model_path}/gpt_infer_past.xml"))
        self.llm_past_compiled_model = core.compile_model(self.llm_past_model, device)# config = {'INFERENCE_PRECISION_HINT': 'f32'})
        self.llm_past_request = self.llm_past_compiled_model.create_infer_request()

        if int4_compress:
            self.llm_model = core.read_model(Path(f"{ov_model_path}/gpt_infer_int4.xml"))
        else:
            self.llm_model = core.read_model(Path(f"{ov_model_path}/gpt_infer.xml"))

        self.llm_compiled_model = core.compile_model(self.llm_model, device) # config = {'INFERENCE_PRECISION_HINT': 'f32'})
        self.llm_request = self.llm_compiled_model.create_infer_request()

        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.llm_model.inputs)}
        self.output_names = {idx: key for idx, key in enumerate(self.llm_model.outputs)}
        self.key_value_input_names = [key for key in list(self.input_names) if key not in ["beam_idx", "inputs_embeds", "attention_mask", "position_ids"]]
        self.key_value_output_names = [key for key in list(self.output_names)[1:]]
        self.stateful = len(self.key_value_input_names) == 0
        # self.compiled_model = core.compile_model(self.model, device, config = {'INFERENCE_PRECISION_HINT': 'f32'})

        self.device = torch.device("cpu")        
        self.model_ov_load()

    def model_ov_load(self):
        hifi_model_path = Path(f"{self.ov_model_path}/hifi.xml")
        gpt_model_path = Path(f"{self.ov_model_path}/gpt.xml")

        hifi_model = self.core.read_model(hifi_model_path)
        gpt_model = self.core.read_model(gpt_model_path)

        self.hifi_model_compiled = self.core.compile_model(hifi_model, self.ov_device)
        self.gpt_model_compiled = self.core.compile_model(gpt_model, self.ov_device)

        self.hifi_model_request = self.hifi_model_compiled.create_infer_request()
        self.gpt_model_request = self.gpt_model_compiled.create_infer_request()

    def hifi_model_run(self, latents=None, speaker_embedding=None):
        inputs_dict = {}
        inputs_dict['latents'] = latents
        inputs_dict['speaker_embedding'] = speaker_embedding

        self.hifi_model_request.start_async(inputs_dict, share_inputs=True)
        self.hifi_model_request.wait()
        
        return torch.from_numpy(self.hifi_model_request.get_tensor("wave_data").data)

    def gpt_model_run(self, text_input=None, text_lengths=None, audio_codes=None, wav_lengths=None, cond_latents=None):
        inputs_dict = {}
        inputs_dict['text_input'] = text_input
        inputs_dict['text_lengths'] = text_lengths
        inputs_dict['audio_codes'] = audio_codes
        inputs_dict['wav_lengths'] = wav_lengths
        inputs_dict['cond_latents'] = cond_latents
        
        self.gpt_model_request.start_async(inputs_dict, share_inputs=True)
        self.gpt_model_request.wait()
        
        return torch.from_numpy(self.gpt_model_request.get_tensor("mel_logits").data)
    
    def gpt_inference_run(self, inputs_embeds=None, attention_mask=None, position_ids=None):
        inputs_dict = {}
        inputs_dict['inputs_embeds'] = inputs_embeds
        inputs_dict['attention_mask'] = attention_mask 
        inputs_dict['position_ids'] = position_ids
        self.llm_request.start_async(inputs_dict, share_inputs=True)
        self.llm_request.wait()
        
        pkv = []
        for idx in range(30):
            item_kv = (torch.from_numpy(self.llm_request.get_tensor(f"present.{idx}.key").data), torch.from_numpy(self.llm_request.get_tensor(f"present.{idx}.value").data))
            pkv.append(item_kv)

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=torch.from_numpy(self.llm_request.get_tensor("lm_logits").data),
            past_key_values=tuple(pkv),
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )
    
    def gpt_inference_past_run(self, inputs_embeds=None, attention_mask=None, position_ids=None, pkv=None):
        inputs_dict = {}
        inputs_dict['inputs_embeds'] = inputs_embeds
        inputs_dict['attention_mask'] = attention_mask 
        inputs_dict['position_ids'] = position_ids

        for idx in range(30):
            inputs_dict[f"past_key_values.{idx}.key"] = pkv[idx][0]
            inputs_dict[f"past_key_values.{idx}.value"] = pkv[idx][1]

        self.llm_past_request.start_async(inputs_dict, share_inputs=True)
        self.llm_past_request.wait()
        
        pkv = []
        for idx in range(30):
            item_kv = (torch.from_numpy(self.llm_past_request.get_tensor(f"present.{idx}.key").data), torch.from_numpy(self.llm_past_request.get_tensor(f"present.{idx}.value").data))
            pkv.append(item_kv)

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=torch.from_numpy(self.llm_past_request.get_tensor("lm_logits").data),
            past_key_values=tuple(pkv),
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )
    
    