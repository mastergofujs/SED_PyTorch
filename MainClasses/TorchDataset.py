from torch.utils.data import Dataset
from MainClasses.DataHandler import DataHandler

'''
This class defines the operations of datasets.
'''


class TorchDataset(Dataset):
    def __init__(self, options, type_, fold):
        super(TorchDataset, self).__init__()
        self.dh = DataHandler(dataset=options.dataset)
        self.ts = options.time_step
        self.K = options.num_events
        datas = self.dh.load_data(nevents=options.num_events, fold=fold)
        self.get_datas(type_=type_, datas=datas)

    def get_datas(self, type_, datas):
        if type_ == 'train':
            self.x_data = self.dh.sequentialize_data(datas[0], timestep=self.ts)
            self.y_data = datas[1][self.ts - 1:]
        elif type_ == 'val':
            self.x_data = self.dh.sequentialize_data(datas[2], timestep=self.ts)
            self.y_data = datas[3][self.ts - 1:]
        elif type_ == 'test':
            self.x_data = self.dh.sequentialize_data(datas[4], timestep=self.ts)
            self.y_data = datas[5][self.ts - 1:]
        else:
            print('ERROR DATA TYPE!')
            return

    def __getitem__(self, item):
        x_data = self.x_data[item]
        y_data = self.y_data[item]
        return x_data, y_data

    def __len__(self):
        return len(self.x_data)
