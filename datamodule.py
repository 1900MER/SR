import lightning as L
from srdataset import YouKuSrDataset
from torch.utils.data import DataLoader, Dataset
from typing import List, Union, Any,Optional



class SRDataModule(L.LightningDataModule):
    def __init__(self,
                 train_transforms = None,
                 val_transforms = None,
                 batch_size  = None,
                 num_workers = None,
                 drop_last = None, 
                 pin_memory = None,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__()
        self.batch_size = batch_size 
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        
    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            self.dataset_train = YouKuSrDataset(split='train',val_video=1,transforms=self.train_transforms)
            self.dataset_val   = YouKuSrDataset(split='val',  val_video=1,transforms=self.val_transforms)
        
    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return self._data_loader(self.dataset_train, shuffle=True)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        return self._data_loader(self.dataset_val)

        