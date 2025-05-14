# Non-IID划分实现 将 MNIST 数据划分为 Non-IID 分布（每客户端2类）

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_non_iid_mnist(num_clients=100, shards_per_client=2, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))
    idxs_labels = np.vstack((idxs, labels)).T
    idxs_labels = idxs_labels[idxs_labels[:, 1].argsort()]

    shards = np.array_split(idxs_labels, num_clients * shards_per_client)
    np.random.shuffle(shards)

    client_data_idx = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        for j in range(shards_per_client):
            client_data_idx[i].extend(shards[i * shards_per_client + j][:, 0])

    loaders = [DataLoader(Subset(dataset, idxs), batch_size=batch_size, shuffle=True)
               for idxs in client_data_idx]
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return loaders, test_loader
