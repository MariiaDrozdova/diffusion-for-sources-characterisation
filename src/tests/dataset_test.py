import unittest
import os
import torch

# Import the necessary classes/functions from your actual code file.
# Assume your code is in a file named `data_loading.py`
from src.data import SkaDataset, MakeDataLoader

class TestSkaDataset(unittest.TestCase):
    def setUp(self):
        # Setup some basic parameters for your tests.
        self.folder_images = './test_data/'  # Assuming you have a test_data folder with sample data
        self.file_labels = None  # Assuming no separate file for labels in your example
        self.image_size = 128

    def test_dataset_length(self):
        dataset = SkaDataset(self.folder_images, self.file_labels, self.image_size, power=10, from_uv=False)
        self.assertTrue(len(dataset) > 0)

    def test_dataset_item(self):
        dataset = SkaDataset(self.folder_images, self.file_labels, self.image_size, power=10, from_uv=False)
        item = dataset[0]
        self.assertIn("true", item)
        self.assertIn("dirty_noisy", item)
        self.assertIsInstance(item["true"], torch.Tensor)
        self.assertIsInstance(item["dirty_noisy"], torch.Tensor)

    def test_make_data_loader(self):
        dataloader = MakeDataLoader(self.folder_images, self.file_labels, self.image_size)
        train_loader = dataloader.get_data_loader_train()
        for batch in train_loader:
            self.assertIn("true", batch)
            self.assertIn("dirty_noisy", batch)
            break  # Testing just one batch for simplicity

if __name__ == '__main__':
    unittest.main()