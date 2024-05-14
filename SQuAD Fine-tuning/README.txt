1. Change the name of the base model to the name (if it is available on Hugging Face) or file path (otherwise) of the model you will be fine-tuning, ensuring that the tokenizer for the model is saved in the same directory as the model.
2. Change the name of the target model as desired
3. Change the WandB API key
4. Change the name of the WandB project the runs will be saved to
5. The training and validation datasets can be loaded from an arrow file or from Hugging Face