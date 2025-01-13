from pathlib import Path
import os
from tests import _PATH_DATA

from torch.utils.data import Dataset, TensorDataset

from my_project.data import MyDataset
from my_project.data import load_processed_data
import pytest

_PATH_DATA = Path(_PATH_DATA)

def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset(_PATH_DATA.joinpath("raw"))
    assert isinstance(dataset, Dataset) # check if dataset is an instance of torch.utils.data.Dataset

testdata = [
    (30000, 5000, (30000, 1, 28, 28), (5000, 1, 28, 28)), # the correct one
    #(30000, 5000, (30000, 784), (5000, 784)),
    #(50000, 5000, (50000, 1, 28, 28), (5000, 1, 28, 28)),
    #(50000, 5000, (50000, 784), (5000, 784)),
]

# @pytest.mark.parametrize("output_folder", [None, "processed"])
@pytest.mark.parametrize("N_train,N_test,train_shape,test_shape", testdata)
def test_load_processed_data(N_train, N_test, train_shape, test_shape):
    train, test = load_processed_data(_PATH_DATA.joinpath("processed"))
    train_img, train_labels = train.tensors
    test_img, test_labels = test.tensors
    assert isinstance(train, TensorDataset)  # type: Tensors
    assert isinstance(test, TensorDataset)   # type: Tensors
    assert len(train) == N_train            # length of train data
    assert len(test) == N_test              # length of test data
    assert train_img.shape == train_shape         # shape of train data
    assert test_img.shape == test_shape         # shape of test data
    assert set(train_labels.tolist()) == set(range(10)) # all lab. represented
    assert set(test_labels.tolist()) == set(range(10))  # all lab. represented


