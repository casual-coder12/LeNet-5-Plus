# LeNet-5 Plus

A TensorFlow/Keras image-classification project based on LeNet-5, extended with additional convolutional capacity, batch normalization, dropout, and data augmentation. The project supports MNIST and CIFAR-10 and includes training, evaluation, and visualization utilities.

## CIFAR-10 Results

The upgraded LeNet-5 Plus model improved CIFAR-10 test accuracy from **53.5% to 76.5%**, a gain of **23 percentage points** (about 43% relative improvement over the baseline).

| Model | CIFAR-10 accuracy |
|---|---:|
| Baseline LeNet-5 | 53.5% |
| Upgraded LeNet-5 Plus | **76.5%** |

This is an improvement of **23 percentage points** over the baseline. The upgraded-model experiment focused on CIFAR-10. MNIST remains supported by the project code, but was left out of this comparison because both the baseline and upgraded models already achieved high accuracy on it.

The reported 50-epoch experiment used a stepwise learning-rate schedule:

| Training phase | Learning rate |
|---|---:|
| Epochs 1–20 | 0.001 |
| Epochs 21–40 | 0.0001 |
| Epochs 41–50 | 0.00001 |

The validation curves show some oscillation. Random data augmentation and dropout may contribute to less smooth training from epoch to epoch. They are active only during training, not validation, so any effect on validation metrics is indirect, through the model weights learned during training. This is a plausible explanation, not a confirmed diagnosis.

**Reproducibility note:** the schedule above describes the reported experiment. The current `train.py` accepts an initial learning rate, but `LeNetTrainer` does not currently implement automatic learning-rate scheduling. Reproducing the exact schedule requires adding a scheduler or changing the learning rate during training.

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
