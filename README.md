```
git clone https://github.com/jeakwon/engram.git -q
cd engram
pip install -e . -q
python -m engram.finetuning.finetune_trainer --config engram/finetuning/config/sgd/cifar10_vit_small_patch16_224pt.yaml
```
