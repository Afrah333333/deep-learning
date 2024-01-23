{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyOoFKWpAKiCciV8Tji7epQR",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Afrah333333/deep-learning/blob/main/model_builder.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\"\"\"\n",
        "Contains PyTorch model code to instantiate a TinyVGG model.\n",
        "\"\"\"\n",
        "import torch\n",
        "from torch import nn\n",
        "\n",
        "class TinyVGG(nn.Module):\n",
        "    \"\"\"Creates the TinyVGG architecture.\n",
        "\n",
        "    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.\n",
        "    See the original architecture here: https://poloclub.github.io/cnn-explainer/\n",
        "\n",
        "    Args:\n",
        "    input_shape: An integer indicating number of input channels.\n",
        "    hidden_units: An integer indicating number of hidden units between layers.\n",
        "    output_shape: An integer indicating number of output units.\n",
        "    \"\"\"\n",
        "    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:\n",
        "        super().__init__()\n",
        "        self.conv_block_1 = nn.Sequential(\n",
        "          nn.Conv2d(in_channels=input_shape,\n",
        "                    out_channels=hidden_units,\n",
        "                    kernel_size=3,\n",
        "                    stride=1,\n",
        "                    padding=0),\n",
        "          nn.ReLU(),\n",
        "          nn.Conv2d(in_channels=hidden_units,\n",
        "                    out_channels=hidden_units,\n",
        "                    kernel_size=3,\n",
        "                    stride=1,\n",
        "                    padding=0),\n",
        "          nn.ReLU(),\n",
        "          nn.MaxPool2d(kernel_size=2,\n",
        "                        stride=2)\n",
        "        )\n",
        "        self.conv_block_2 = nn.Sequential(\n",
        "          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),\n",
        "          nn.ReLU(),\n",
        "          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),\n",
        "          nn.ReLU(),\n",
        "          nn.MaxPool2d(2)\n",
        "        )\n",
        "        self.classifier = nn.Sequential(\n",
        "          nn.Flatten(),\n",
        "          # Where did this in_features shape come from?\n",
        "          # It's because each layer of our network compresses and changes the shape of our inputs data.\n",
        "          nn.Linear(in_features=hidden_units*13*13,\n",
        "                    out_features=output_shape)\n",
        "        )\n",
        "\n",
        "    def forward(self, x: torch.Tensor):\n",
        "        x = self.conv_block_1(x)\n",
        "        x = self.conv_block_2(x)\n",
        "        x = self.classifier(x)\n",
        "        return x\n",
        "        # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion"
      ],
      "metadata": {
        "id": "ZI24WSylAiYw"
      },
      "execution_count": 5,
      "outputs": []
    }
  ]
}