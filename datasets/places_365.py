from torch.utils.data import Dataset
from torchvision.datasets import Places365 as Places365Original

class Places365(Dataset):
    """
    Places365 dataset wrapper.
    """
    def __init__(self, root, train: bool = True,
                 transform=None, download: bool = False):
        super().__init__()

        split = "train-standard" if train else "val"

        self.places = Places365Original(
            root=root,
            split=split,
            small=True,
            transform=transform,
            download=download
        )

    def __len__(self):
        return len(self.places)

    def __getitem__(self, index):
        image, class_id = self.places[index]  
        return image, class_id