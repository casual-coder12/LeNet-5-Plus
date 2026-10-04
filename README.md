# LeNet-5 Plus

A TensorFlow/Keras image-classification project based on LeNet-5, extended with additional convolutional capacity, batch normalization, dropout, and data augmentation. The project supports MNIST and CIFAR-10 and includes training, evaluation, and visualization utilities.

## CIFAR-10 Results

The upgraded LeNet-5 Plus model improved CIFAR-10 test accuracy from **53.5% to 76.5%**, a gain of **23 percentage points** (about 43% relative improvement over the baseline).

| Model | CIFAR-10 accuracy |
|---|---:|
| Baseline LeNet-5 | 53.5% |
| Upgraded LeNet-5 Plus | **76.71%** |

This is an improvement of **23 percentage points** over the baseline. The upgraded-model experiment focused on CIFAR-10. MNIST remains supported by the project code, but was left out of this comparison because both the baseline and upgraded models already achieved high accuracy on it.

Training starts with a learning rate of `0.001` and uses Keras `ReduceLROnPlateau`, monitoring validation loss (`val_loss`). If validation loss does not improve for 3 consecutive epochs, the callback halves the learning rate, down to a minimum of `0.000001`. Since each reduction depends on validation-loss behavior, the learning rate does not follow a fixed epoch-by-epoch schedule.

The validation curves oscillated more at the beginning of training and stabilized afterward. Data augmentation and dropout may contribute to the early fluctuations, but this is not confirmed. Both are active during training rather than validation, so their influence on validation metrics is indirect, through the learned model weights.

## Model

The model accepts 32×32 grayscale MNIST images or 32×32 RGB CIFAR-10 images. Its main components are:

- Random crop, horizontal flip, rotation, and translation augmentation during training
- Three convolutional layers with 32, 64, and 128 filters, each followed by batch normalization and ReLU
- Two max-pooling layers
- Flattening, dropout (rate 0.5), and an 84-unit ReLU dense layer
- A 10-unit softmax output layer

## Installation

```bash
pip install -r requirements.txt
pip install datasets
```

The additional `datasets` package is used to load CIFAR-10 in `data/dataset.py` and is not currently listed in `requirements.txt`.

## Usage

Train on CIFAR-10:

```bash
python train.py --dataset cifar10 --epochs 50 --batch_size 64 --learning_rate 0.001
```

Train on MNIST:

```bash
python train.py --dataset mnist --epochs 15 --batch_size 64
```

`train.py` supports `--dataset` (`mnist` or `cifar10`), `--epochs`, `--batch_size`, `--learning_rate`, `--save_type` (`w`, `m`, or `both`), and `--load_checkpoint`. Defaults are MNIST, 15 epochs, batch size 64, learning rate 0.001, and saving both model weights and the full model.

The evaluation script accepts `--dataset` and `--load_type` (`w` for weights, `m` for a full model, or `b` for the best checkpoint).

## Project Structure

```text
.
├── data/             # Dataset loading and preprocessing
├── models/           # LeNet-5 Plus model definition and model loader
├── utils/            # Training and visualization utilities
├── notebooks/        # Interactive Jupyter notebook
├── saved_models/     # Saved models, weights, and training history
├── outputs/          # Generated plots and evaluation visualizations
├── train.py          # Training entry point
├── evaluate.py       # Evaluation entry point
└── requirements.txt  # Python dependencies
```

The notebook is `notebooks/lenet5Plus.ipynb`. Generated artifacts include model files (`.keras` and `.weights.h5`), per-epoch history CSV files, training curves, confusion matrices, and sample prediction plots.

## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). “Gradient-based learning applied to document recognition.” *Proceedings of the IEEE*, 86(11), 2278–2324.
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)

## License

Provided for educational purposes.
