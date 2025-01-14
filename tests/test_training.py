# this works!
import os

import torch
from tests.simple_training_loop import train

import pytest

def test_train(mocker):
  mocker.patch('tests.simple_training_loop.torch.save')
  train(lr=1e-3, batch_size=32, epochs=1, print_every=100)
  torch.save.assert_called_once()

if __name__ == "__main__":
  test_train()

