import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from model.DSTANet import make_model
from lib.utils import load_graphdata_channel1, get_adjacency_matrix, compute_val_loss_DSTANet, \
    predict_and_save_results_DSTANet
from tensorboardX import SummaryWriter
from lib.metrics import masked_mape_np, masked_mae, masked_mse, masked_rmse

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='./configurations/PEMS04_DSTANet.conf', type=str,
                    help='configuration file path')
args = parser.parse_args()

config = configparser.ConfigParser()
print('Read configuration file: %s' % args.config)
config.read(args.config,encoding='utf-8')

data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']

model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
# 自动选择合适的 time_strides
if num_of_hours > 0:
    time_strides = num_of_hours
elif num_of_days > 0:
    time_strides = num_of_days
elif num_of_weeks > 0:
    time_strides = num_of_weeks
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])
delay = int(training_config['delay'])

folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (
    model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
# print('folder_dir:', folder_dir)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('params_path:', params_path)

train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = \
    load_graphdata_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size)

adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                 num_for_predict, len_input, num_of_vertices, delay) 


def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % params_path)
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % params_path)
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)
    masked_flag = 0
    criterion = nn.L1Loss().to(DEVICE)
    criterion_masked = masked_mae
    if loss_function == 'masked_mse':
        criterion_masked = masked_mse  # nn.MSELoss().to(DEVICE)
        masked_flag = 1

    elif loss_function == 'masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag = 0
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()

    if start_epoch > 0:
        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    # train model
    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        if masked_flag:
            val_loss = compute_val_loss_DSTANet(net, val_loader, criterion_masked, masked_flag, missing_value, sw, epoch)
        else:
            val_loss = compute_val_loss_DSTANet(net, val_loader, criterion, masked_flag, missing_value, sw, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)

        net.train()  # ensure dropout layers are in train mode

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, labels = batch_data

            optimizer.zero_grad()

            outputs = net(encoder_inputs)

            if masked_flag:
                loss = criterion_masked(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

            # 每100步输出一次训练进度
            if global_step % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_index+1}/{len(train_loader)}], '
                    f'Training Loss: {training_loss:.4f}, Time: {time() - start_time:.2f}s')

            if global_step % 1000 == 0:
                print('global step: %s, training loss: %.2f, time: %.2fs' % (
                    global_step, training_loss, time() - start_time))

        # 每个 epoch 结束时强制输出一次训练进度
        print(f'End of Epoch [{epoch+1}/{epochs}], Final Step [{len(train_loader)}/{len(train_loader)}], '
            f'Training Loss: {training_loss:.4f}, Time: {time() - start_time:.2f}s')

    print('best epoch:', best_epoch)


    # apply the best model on the test set
    predict_main(best_epoch, test_loader, test_target_tensor, metric_method, _mean, _std, 'test')


def predict_main(global_step, data_loader, data_target_tensor, metric_method, _mean, _std, type):
    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results_DSTANet(net, data_loader, data_target_tensor, global_step, metric_method, _mean, _std,
                                    params_path, type)


if __name__ == "__main__":
    train_main()
