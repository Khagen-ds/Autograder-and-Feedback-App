import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class FeedbackMaker:
    """
    Wrapper for Qwen 2.5-3B-Instruct (can be changed) to generate feedback from prompts classes (see LLM_feedback folder)
    """

    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", device="cpu"):
        self.device = device    #    Can be changed to gpu later if available, but currently CPU

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,   # CPU-safe
            low_cpu_mem_usage=True
        )

        self.model.eval()

    def make_feedback(self, prompt, max_new_tokens=1200):
        """
        Make feedback from prompt
        """

        # Use chat template if available, to automatically format input into: system instructions -> prompt from user -> Response from LLM
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            )
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move all input tensors to the same device(CPU, or GPU in the future)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,    #    False = stable/ consistent, True = more natural/ varied
                #temperature=0.7,    #    remove # if do_sample = True
                #top_p=0.9,         #    remove # if do_sample = True
                pad_token_id=self.tokenizer.eos_token_id    #    Prevent edge-case issues
            )

        input_length = inputs["input_ids"].shape[1]

        output_text = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

        return output_text
