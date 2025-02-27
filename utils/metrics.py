import torch

class MeanMetric:
    def __init__(self):
        self.reset()

    def update(self, value):
        value = value[~torch.isnan(value)]
        self.sum += value
        self.count += value.shape[0]

    def compute(self):
        """
        compute current mean metric.
        """
        if self.count == 0:
            return torch.tensor(0.0)
        return self.sum / self.count

    def reset(self):
        """
        reset metric.
        """
        self.sum = torch.tensor(0.0)
        self.count = torch.tensor(0.0)

class BinaryAccuracy:
    def __init__(self):
        self.reset()

    def update(self, out, pred):
        filter_index = ~torch.isnan(out) * ~torch.isnan(pred)
        out = out[filter_index]
        pred = pred[filter_index]
        self.out = torch.cat([self.out, out], dim=0)
        self.pred = torch.cat([self.pred, pred], dim=0)
        self.count += out.shape[0]

    def compute(self):
        """
        compute current binary accuracy.
        """
        if self.count == 0:
            return torch.tensor(0.0)
        return (self.out == self.pred).sum().item() / self.count

    def reset(self):
        """
        reset metric.
        """
        self.out = torch.tensor([])
        self.pred = torch.tensor([])
        self.count = torch.tensor(0.0)


if __name__ == "__main__":
    metric = MeanMetric()
    metric.update(5)
    metric.update(10, n=2)
    print("current mean metric:", metric.compute())  # output: current mean metric: 5.0
    metric.reset()
    print("mean after reset:", metric.compute())  #output: mean after reset: 0.0