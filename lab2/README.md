# ![](icon.png) AutoCoder
you can generate cpp java python 

#### Fine-tuning yours
```
git clone https://github.com/AsVoider/coder.git
conda install <requirement.txt> ---- I've modified it , it now contains all needed environment.
```

1. Preparing [the dataset](./dataset)
2. Start fine-tuning model: `python train.py --model_select distilgpt2` 
3. After fine-tuning, the model will be saved to `./tmp/model/distilgpt2_fine_tuned_coder/0_GPTSingleHead`
4. Move ./tmp/model/distilgpt2_fine_tuned_coder/0_GPTSingleHead to ./model before another training
5. I've changed the path to find models, if need a brand new model, just modify train.py file.
6. It can support more languages, just download and prepare in ./dataset
7. I've also changed the logic in trainer.py; when you start a training, it will nolonger ask u if you want to remove ./tmp; just for running on kaggle
8. ./p+j 是训练Java，python4轮后的log， ./p_j_c是训练cpp,java,python3轮后的log

