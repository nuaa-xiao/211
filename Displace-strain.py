import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader

from model import Displace_Strain
import config
from lib.dataset.FEMdata import load_data_from_directory, split_data, repeat_tensor_elements, CustomDataset, load_data_from_directory1

def executeEpoch(model, loader, loss_function, opt, sched, e, sw, mode='train'):
    assert mode == 'train' or mode =='val', 'mode should be train or val'
    lE = 0.0
    if mode == 'train':
        model.train()
    else:
        model.eval()

    for b, (input1, input2, displace, strain) in enumerate(loader):
        input1 = input1.cuda()
        input2 = input2.cuda()
        displace = displace.cuda()
        strain = strain.cuda()
        if mode =='train':
            opt.zero_grad()
            output_displacement, output_strain = model(input1, input2)
            loss1 = loss_function(displace,strain,output_displacement,output_strain)
            loss1.backward()
            opt.step()
            opt.zero_grad()
            print(f'Epoch {e}/B {b}. Loss {loss1.item()}')
        lE+=loss1.item()
    lE /= len(loader)
    sw.add_scalar(f'{mode}/loss', lE, e)
    if mode == 'train':
        sched.step()

def train(args):
    input1_data = load_data_from_directory(config.INPUT_PATH,needsorted=True)
    input2_data = load_data_from_directory(config.COORD_PATH,needsorted=False)
    displace = load_data_from_directory(config.OUTPUT_PATH,needsorted=True)
    strain = load_data_from_directory(config.OUTPUT_PATH1,needsorted=True)

    input1_data = torch.Tensor(input1_data)
    input2_data = torch.Tensor(input2_data)
    displace = torch.Tensor(displace)
    strain = torch.Tensor(strain)

    assert all(isinstance(t, torch.Tensor) for t in input1_data)
    assert all(isinstance(t, torch.Tensor) for t in input2_data)
    assert all(isinstance(t, torch.Tensor) for t in displace)
    assert all(isinstance(t, torch.Tensor) for t in strain)

    dataset = CustomDataset(input1_data, input2_data, displace, strain)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, pin_memory=True, drop_last=True, num_workers=config.batch_size, shuffle=True)

    model = Displace_Strain().cuda()
    loss_fn = torch.nn.L1Loss(reduction='mean')
    opt = torch.optim.Adam(model.parameters(), lr=1e-1, eps=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    print(model)

    expPath = 'runs/'
    writer = SummaryWriter(expPath)

    for e in range(config.epochs):
        executeEpoch(model, dataloader, loss_fn, opt, sched, e, writer, mode='train')
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains FastReg registration model')
    args = parser.parse_args()
    train(args)