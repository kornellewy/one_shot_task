import unittest
import pandas as pd
import random
import torch
import torchvision.transforms as transforms
import cv2

from dataset import AlbumentationsTransform, ImbalancedDatasetSampler, FashionMnist, AlbumentationsTransformTest
from model_ver001 import FashionCNNVer001
from model_ver002 import FashionCNNVer002


class TestFashionMnist(unittest.TestCase):
    def setUp(self):
        self.train_dataset_path = 'Fashion_MNIST/fashion-mnist_train.csv'
        self.train_dataset = pd.read_csv(self.train_dataset_path)
        self.test_dataset_path = 'Fashion_MNIST/fashion-mnist_test.csv'
        self.test_dataset = pd.read_csv(self.test_dataset_path)

    def test_dataset_len(self):
        dataset = FashionMnist(self.train_dataset)
        self.assertEqual(len(dataset), 60000)
        dataset = FashionMnist(self.test_dataset)
        self.assertEqual(len(dataset), 10000)

    def test_getitem(self):
        dataset = FashionMnist(self.train_dataset)
        idx = random.randint(0, len(dataset)-1)
        for i in range(100):
            image, label = dataset[idx]
            self.assertEqual(image.shape, (28, 28, 1))
            self.assertEqual(label.shape, ())
            self.assertEqual(label==0 or label==1, True)


class TestFashionCNNVer001(unittest.TestCase):
    def test_forward_and_output_shape(self):
        model = FashionCNNVer001()
        for i in range(100):
            input_tensor = torch.rand(1, 1, 28, 28)
            output_tensor = model.forward(input_tensor)
            self.assertEqual(output_tensor.shape, torch.Size([1, 2]))


class TestAlbumentationsTransform(unittest.TestCase):
    def setUp(self):
        self.test_image_path = 'test_img.jpg'

    def test_agumentations(self):
        image = cv2.imread(self.test_image_path)
        test_class = AlbumentationsTransform()
        agu_image = test_class(image)
        self.assertEqual(image.all()==None, False)
        self.assertEqual(agu_image['image'].all() == None, False)
        self.assertEqual(self.is_similar(agu_image['image'], image), False)

    def is_similar(self, image1, image2):
        return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())

class TestAlbumentationsTransformTest(unittest.TestCase):
    def setUp(self):
        self.test_image_path = 'test_img.jpg'

    def test_agumentations(self):
        image = cv2.imread(self.test_image_path)
        test_class = AlbumentationsTransformTest()
        agu_image = test_class(image)
        self.assertEqual(image.all()==None, False)
        self.assertEqual(agu_image['image'].all() == None, False)
        self.assertEqual(self.is_similar(agu_image['image'], image), False)

    def is_similar(self, image1, image2):
        return image1.shape == image2.shape and np.bitwise_xor(image1,image2).any()


class TestImbalancedDatasetSampler(unittest.TestCase):
    def setUp(self):
        self.dataset_path = 'Fashion_MNIST/fashion-mnist_train.csv'
        self.dataset = pd.read_csv(self.dataset_path)
        self.img_transforms = transforms.Compose([AlbumentationsTransform()])
        self.dataset = FashionMnist(self.dataset, self.img_transforms)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                            sampler=ImbalancedDatasetSampler(self.dataset), batch_size=32)

    def test_batch_class_balance(self):
        class_0_count = 0
        class_1_count = 0
        for i in range(10):
            for images, labels in self.dataloader:
                for label in labels:
                    if label.item() == 0:
                        class_0_count+=1
                    else:
                        class_1_count+=1
        sum_count = class_0_count + class_1_count
        self.assertEqual(class_0_count/sum_count > 0.45, True)
        self.assertEqual(class_1_count/sum_count > 0.45, True)


class TestFashionCNNVer002(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.base_model_path = 'models/test_loss_0.005_accuracy_0.9984.pth'
        self.base_model = torch.load(self.base_model_path).to(self.device)
    
    def test_model002_input_output_shape(self):
        input_tensor = torch.rand(1, 1, 28, 28)
        model = FashionCNNVer002(base_model=self.base_model).to(self.device)
        output_tensor = model.forward(input_tensor, input_tensor, input_tensor)
        self.assertEqual(output_tensor.shape, torch.Size([1]))


if __name__ == '__main__':
    unittest.main()