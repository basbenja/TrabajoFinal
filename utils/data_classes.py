from torch.utils.data import Dataset

class TemporalStaticDataset(Dataset):
    def __init__(self, temporal_data, static_data, labels):
        """
        Args:
            temporal_data (torch.Tensor): Tensor of shape (num_samples, seq_len, input_dim).
            static_data (torch.Tensor): Tensor of shape (num_samples, static_dim).
            labels (torch.Tensor): Tensor of shape (num_samples,).
        """
        self.temporal_data = temporal_data
        self.static_data = static_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (temporal_data, static_data, label)
        """
        temporal_sample = self.temporal_data[idx]
        static_sample = self.static_data[idx]
        label = self.labels[idx]
        return temporal_sample, static_sample, label