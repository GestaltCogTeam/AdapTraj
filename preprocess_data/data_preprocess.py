import os
import csv
import glob
import yaml
import json
import math
import torch
import numpy as np
import pandas as pd
import pickle
from scipy.interpolate import interp1d


def change_file_into_csv():
    """
    default fps = 2.5, x,y in meters
    Returns: csv file whose line is [frame, agentID, x, y]
    """
    datasets_name = ['ETH', 'UCY', 'L-CAS', 'SDD', 'WildTrack', 'SYI', 'CFF']
    datasets_name = ['SDD']
    for dataset_name in datasets_name:
        data_path = f'../new_datasets/raw/{dataset_name}'
        csv_data_path = f'../new_datasets/csv_raw/{dataset_name}'
        os.makedirs(csv_data_path, exist_ok=True)
        datasets_list = os.listdir(data_path)
        if dataset_name == 'ETH':
            # each file is captured at different scene, fps=2.5
            for file in datasets_list:
                data = []
                with open(os.path.join(data_path, file), 'r') as f:
                    for line in f:
                        line = line.strip().split(' ')
                        line = [float(i) for i in line if i != '']
                        # line = [frame agentID pos_x pos_z pos_y v_x v_z v_y]
                        data.append([int(float(line[0])-1), int(float(line[1])), float(line[2]), float(line[4])])
                # data = [frame, agentID, x, y]
                data = np.asarray(data)

                csv_file_name = file[:-3] + 'csv'
                csv_file = open(os.path.join(csv_data_path, csv_file_name), 'w')
                csv_file.write("frame, agentID, x, y\n")
                np.savetxt(csv_file, data, delimiter=',')
                csv_file.close()
                print(f'change raw file into csv file in {file}')
        elif dataset_name == 'UCY':
            # each file is captured at different scene
            # interpolate to achieve fps=2.5
            for file in datasets_list:
                data = []

                pedestrians = []
                current_pedestrian = []

                with open(os.path.join(data_path, file), 'r') as f:
                    for line in f:
                        line = line.split('\n')[0]
                        if ('- Num of control points' in line) or ('- the number of splines' in line):
                            if current_pedestrian:
                                pedestrians.append(current_pedestrian)
                            current_pedestrian = []
                            continue

                        # strip comments
                        if ' - ' in line:
                            line = line[:line.find(' - ')]

                        # tokenize
                        entries = [e for e in line.split(' ') if e]
                        if len(entries) != 4:
                            continue

                        x, y, f, _ = entries
                        current_pedestrian.append([float(x), float(y), int(f)])

                    if current_pedestrian:
                        pedestrians.append(current_pedestrian)

                for ped_id, person_xyf in enumerate(pedestrians):
                    ## Earlier
                    # xs = np.array([x for x, _, _ in person_xyf]) / 720 * 12 # 0.0167
                    # ys = np.array([y for _, y, _ in person_xyf]) / 576 * 12 # 0.0208

                    ## Pixel-to-meter scale conversion according to
                    ## https://github.com/agrimgupta92/sgan/issues/5
                    xs = np.array([x for x, _, _ in person_xyf]) * 0.0210
                    ys = np.array([y for _, y, _ in person_xyf]) * 0.0239

                    fs = np.array([f for _, _, f in person_xyf])

                    kind = 'linear'
                    if len(fs) > 5:
                        kind = 'cubic'

                    x_fn = interp1d(fs, xs, kind=kind)
                    y_fn = interp1d(fs, ys, kind=kind)

                    frames = np.arange(min(fs) // 10 * 10 + 10, max(fs), 10)

                    for x, y, f in np.stack([x_fn(frames), y_fn(frames), frames]).T:
                        data.append([int(f), ped_id, x, y])

                # data = [frame, agentID, x, y]
                data = np.asarray(data)

                csv_file_name = file[:-3] + 'csv'
                csv_file = open(os.path.join(csv_data_path, csv_file_name), 'w')
                csv_file.write("frame, agentID, x, y\n")
                np.savetxt(csv_file, data, delimiter=',')
                # csv_writer = csv.writer(csv_file)
                # csv_writer.writerow(data)
                csv_file.close()
                print(f'change raw file into csv file in {file}')
        elif dataset_name == 'SDD':
            # fps = 30, 12 frames sampled to achieve fps=2.5
            # load the homography values
            with open(os.path.join(data_path, 'estimated_scales.yaml'), 'r') as hf:
                scales_yaml_content = yaml.load(hf, Loader=yaml.FullLoader)
            scene_names = ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'little', 'nexus', 'quad']
            for scene_name in scene_names:
                scene_data_path = os.path.join(data_path, scene_name)
                scene_datasets_list = os.listdir(scene_data_path)
                for scene_video_id in scene_datasets_list:
                    certainty = scales_yaml_content[scene_name][scene_video_id]['certainty']
                    if certainty != 1.0:
                        continue

                    scale = scales_yaml_content[scene_name][scene_video_id]['scale']
                    data = []
                    with open(os.path.join(scene_data_path, scene_video_id, 'annotations.txt'), 'r') as f:
                        for line in f:
                            line = line.strip().split(' ')
                            # line = [agentID x_min y_min x_max y_max frame lost occluded generated label]
                            frame = int(line[5])
                            # label = line[9]  # pedestrians, bikers, skateboarders, cars, buses, and golf carts
                            if frame % 12 == 0:
                                agentID = int(float(line[0]))
                                x = scale * (float(line[1]) + float(line[3])) / 2
                                y = scale * (float(line[2]) + float(line[4])) / 2
                                data.append([frame, agentID, x, y])
                    # data = [frame, agentID, x, y]
                    data = np.asarray(data)

                    csv_file_name = f'{scene_name}-{scene_video_id}.csv'
                    csv_file = open(os.path.join(csv_data_path, csv_file_name), 'w')
                    csv_file.write("frame, agentID, x, y\n")
                    np.savetxt(csv_file, data, delimiter=',')
                    csv_file.close()
                    print(f'change raw file into csv file in {csv_file_name}')
        elif dataset_name == 'L-CAS':
            for file in datasets_list:
                csv_data = pd.read_csv(os.path.join(data_path, file))
                data = np.asarray(csv_data)
                csv_file = open(os.path.join(csv_data_path, file), 'w')
                csv_file.write("frame, agentID, x, y\n")
                np.savetxt(csv_file, data[:, :4], delimiter=',')
                csv_file.close()
                print(f'change raw file into csv file in {file}')
        elif dataset_name == 'WildTrack':
            # FPS = 2, which is different from other datasets
            data = []
            for file in datasets_list:
                frame = int(os.path.basename(file).replace('.json', ''))
                with open(os.path.join(data_path, file), 'r') as f:
                    for entry in json.load(f):
                        ped_id = float(entry['personID'])
                        position_id = float(entry['positionID'])

                        x = -3.0 + 0.025 * (position_id % 480)
                        y = -9.0 + 0.025 * (position_id / 480)

                        data.append([frame, ped_id, x, y])
            data = np.asarray(data)
            csv_file_name = 'wildtrack_data.csv'
            csv_file = open(os.path.join(csv_data_path, csv_file_name), 'w')
            csv_file.write("frame, agentID, x, y\n")
            np.savetxt(csv_file, data, delimiter=',')
            csv_file.close()
            print(f'change raw file into csv file in {csv_file_name}')
        elif dataset_name == 'CFF':
            for file in datasets_list:
                piw_data, pie_data = [], []
                print(f'start to change {file}')
                with open(os.path.join(data_path, file), 'r') as f:
                    for line in f:
                        line = [e for e in line.split(';') if e != '']
                        ## Time Stamp
                        time = [t for t in line[0].split(':') if t != '']
                        ## Check Line Entry Valid
                        if len(line) != 5:
                            return None
                        ## Check Time Entry Valid
                        if len(time) != 4:
                            return None
                        ## Check Time Format
                        if time[0][-3:] == 'T07':
                            ped_id = int(line[4])
                            f = 0
                        elif time[0][-3:] == 'T17':
                            ped_id = 100000 + int(line[4])
                            f = 100000
                        else:
                            # "Time Format Incorrect"
                            return None
                        ## Extract Frame
                        f += int(time[-3]) * 1000 + int(time[-2]) * 10 + int(time[-1][0])
                        if f % 4 == 0:
                            if line[1] == 'PIW':
                                piw_data.append([f, ped_id, float(line[2])/1000, float(line[3])/1000])
                            elif line[1] == 'PIE':
                                pie_data.append([f, ped_id, float(line[2])/1000, float(line[3])/1000])
                            else:
                                print('wrong location')
                piw_data = np.asarray(piw_data)
                csv_file_name = f'cff_{file[11:-4]}_piw_data.csv'
                csv_file = open(os.path.join(csv_data_path, csv_file_name), 'w')
                csv_file.write("frame, agentID, x, y\n")
                np.savetxt(csv_file, piw_data, delimiter=',')
                csv_file.close()
                print(f'change raw file into csv file in {csv_file_name}')

                pie_data = np.asarray(pie_data)
                csv_file_name = f'cff_{file[11:-4]}_pie_data.csv'
                csv_file = open(os.path.join(csv_data_path, csv_file_name), 'w')
                csv_file.write("frame, agentID, x, y\n")
                np.savetxt(csv_file, pie_data, delimiter=',')
                csv_file.close()
                print(f'change raw file into csv file in {csv_file_name}')
        elif dataset_name == 'SYI':
            # FPS = 25, while input rows are sampled every 20 samples
            data = []
            for file in datasets_list:
                track_id = int(os.path.basename(file).replace('.txt', ''))
                chunk = []
                with open(os.path.join(data_path, file), 'r') as f:
                    for line in f:
                        line = line.split('\n')[0]
                        if not line:
                            continue
                        chunk.append(int(line))
                        if len(chunk) < 3:
                            continue
                        # rough approximation of mapping to world coordinates (main concourse is 37m x 84m)
                        data.append([chunk[2], track_id, chunk[0] * 30.0 / 1920, chunk[1] * 70.0 / 1080])
                        chunk = []
            data = np.asarray(data)
            csv_file_name = 'syi_data.csv'
            csv_file = open(os.path.join(csv_data_path, csv_file_name), 'w')
            csv_file.write("frame, agentID, x, y\n")
            np.savetxt(csv_file, data, delimiter=',')
            csv_file.close()
            print(f'change raw file into csv file in {csv_file_name}')
        else:
            print('wrong dataset name')
    print('all datasets are changed into csv file')


def compare_txt_csv_file():
    dataset_name = 'SDD'
    delim = ' '
    data_path = f'../datasets/raw/{dataset_name}'
    csv_data_path = f'../datasets/csv_raw/{dataset_name}'
    datasets_list = os.listdir(data_path)
    for file in datasets_list:
        txt_data = []
        with open(os.path.join(data_path, file), 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                txt_data.append(line)
        # txt_data = [frame, agentID, x, y]
        txt_data = np.asarray(txt_data)

        csv_file_name = file[:-3] + 'csv'
        csv_data = pd.read_csv(os.path.join(csv_data_path, csv_file_name))
        csv_data = np.asarray(csv_data)
        print('compare')


def split_train_val_test_data(train_fraction, val_fraction, dataset_name):
    data_path = f'./new_datasets/csv_raw/{dataset_name}'
    print(f'start to split file {dataset_name}')
    split_data_path = f'./new_datasets/csv_raw_split/{train_fraction}-{val_fraction}/{dataset_name}'
    os.makedirs(split_data_path, exist_ok=True)
    datasets_list = os.listdir(data_path)
    for csv_file in datasets_list:
        data = pd.read_csv(os.path.join(data_path, csv_file))
        data = np.asarray(data)
        data = data[:, :4]   # only frame, agentID, x, y are needed
        frames = sorted(np.unique(data[:, 0]))
        train_split_index = int(len(frames) * train_fraction)
        val_split_index = train_split_index + int(len(frames) * val_fraction)
        train_data = []
        for train_frame in frames[:train_split_index]:
            train_data.append(data[train_frame == data[:, 0], :])
        csv_file_name = csv_file[:-4] + '-train.csv'
        csv_file_stream = open(os.path.join(split_data_path, csv_file_name), 'w')
        csv_file_stream.write("frame, agentID, x, y\n")
        np.savetxt(csv_file_stream, np.concatenate(train_data), delimiter=',')
        csv_file_stream.close()

        val_data = []
        for val_frame in frames[train_split_index:val_split_index]:
            val_data.append(data[val_frame == data[:, 0], :])
        csv_file_name = csv_file[:-4] + '-val.csv'
        csv_file_stream = open(os.path.join(split_data_path, csv_file_name), 'w')
        csv_file_stream.write("frame, agentID, x, y\n")
        np.savetxt(csv_file_stream, np.concatenate(val_data), delimiter=',')
        csv_file_stream.close()

        test_data = []
        for test_frame in frames[val_split_index:]:
            test_data.append(data[test_frame == data[:, 0], :])
        csv_file_name = csv_file[:-4] + '-test.csv'
        csv_file_stream = open(os.path.join(split_data_path, csv_file_name), 'w')
        csv_file_stream.write("frame, agentID, x, y\n")
        np.savetxt(csv_file_stream, np.concatenate(test_data), delimiter=',')
        csv_file_stream.close()
    

def load_data(train_fraction, val_fraction, dataset_names, phase, obs_len, pred_len, batch_size, process_flag='frame',
              rotate_flag='False', reverse_flag='False'):

    data_loaders = {}  # key: data_name,  value: data_loader
    for dataset_name in dataset_names:
        preprocessed_data_path = f'./new_datasets/csv_preprocessed/{train_fraction}-{val_fraction}/{dataset_name}'
        split_data_path = f'./new_datasets/csv_raw_split/{train_fraction}-{val_fraction}/{dataset_name}'
        if os.path.exists(split_data_path) is False:
            split_train_val_test_data(train_fraction, val_fraction, dataset_name)
        datasets_list = os.listdir(split_data_path)
        datasets_list = list(filter(lambda f: f.endswith(f'{phase}.csv'), datasets_list))

        for i_dt, csv_file in enumerate(datasets_list):
            print("{} / {} - loading {}/{}".format(i_dt + 1, len(datasets_list), split_data_path, csv_file))

            dset = TrajectoryDataset(
                split_data_path,
                preprocessed_data_path,
                csv_file,
                phase,
                obs_len=obs_len,
                pred_len=pred_len,
                process_flag=process_flag,
                rotate_flag=rotate_flag,
                reverse_flag=reverse_flag)

            if len(dset) == 0:
                continue

            loader = torch.utils.data.DataLoader(
                dset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=seq_collate)

            data_loaders[dataset_name + '-' + csv_file[:-4]] = loader

    return data_loaders


def old_load_data(train_fraction, val_fraction, dataset_names, phase, obs_len, pred_len, batch_size, process_flag='frame',
              rotate_flag='False', reverse_flag='False'):

    data_loaders = {}  # key: data_name,  value: data_loader
    for dataset_name in dataset_names:
        preprocessed_data_path = f'./datasets/processed/{dataset_name}/{phase}'
        split_data_path = f'./datasets/original/{dataset_name}/{phase}'
        datasets_list = os.listdir(split_data_path)
        datasets_list = list(filter(lambda f: f.endswith('.txt'), datasets_list))

        for i_dt, csv_file in enumerate(datasets_list):
            print("{} / {} - loading {}/{}".format(i_dt + 1, len(datasets_list), split_data_path, csv_file))

            dset = TrajectoryDataset(
                split_data_path,
                preprocessed_data_path,
                csv_file,
                phase,
                obs_len=obs_len,
                pred_len=pred_len,
                process_flag=process_flag,
                rotate_flag=rotate_flag,
                reverse_flag=reverse_flag)

            if len(dset) == 0:
                continue

            loader = torch.utils.data.DataLoader(
                dset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=seq_collate)

            data_loaders[csv_file[:-4]] = loader

    return data_loaders


def seq_collate(data):
    (seq_traj, obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in seq_traj]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    # input format: batch, seq_len, input_size
    traj = torch.cat(seq_traj, dim=0).permute(0, 2, 1)
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(0, 2, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(0, 2, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(0, 2, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(0, 2, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    # loss_mask = torch.cat(loss_mask_list, dim=0)
    # loss_mask = loss_mask_list
    loss_mask_numpy = np.zeros((traj.shape[0], traj.shape[0]))
    init = 0
    for mask in loss_mask_list:
        loss_mask_numpy[init:init+mask.shape[0], init:init+mask.shape[0]] = 1
        init = init+mask.shape[0]
    loss_mask = torch.from_numpy(loss_mask_numpy).type(torch.float)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        traj, obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end
    ]

    return tuple(out)


def analyze_dataset(train_loaders, val_loaders, test_loaders):
    dict_loaders = {'train': train_loaders, 'val': val_loaders, 'test': test_loaders}
    num_of_sequence = []
    num_of_agents_in_each_sequence = []
    velocity_x_of_agents_in_each_sequence = []
    velocity_y_of_agents_in_each_sequence = []
    acc_x_of_agents_in_each_sequence = []
    acc_y_of_agents_in_each_sequence = []
    for loaders in dict_loaders.values():
        for loader in loaders.values():
            num_of_sequence.append(loader.dataset.num_seq)
            for index in loader.dataset.seq_start_end:
                start, end = index
                vel_sequence = loader.dataset.obs_traj_rel[start:end, :, 1:]
                acc_sequence = vel_sequence[:, :, 1:] - vel_sequence[:, :, :-1]
                num_of_agents_in_each_sequence.append(end-start)
                velocity_x_of_agents_in_each_sequence.append(torch.abs(vel_sequence[:, 0, :]).mean())
                velocity_y_of_agents_in_each_sequence.append(torch.abs(vel_sequence[:, 1, :]).mean())
                acc_x_of_agents_in_each_sequence.append(torch.abs(acc_sequence[:, 0, :]).mean())
                acc_y_of_agents_in_each_sequence.append(torch.abs(acc_sequence[:, 1, :]).mean())

    sum_num_of_sequence = torch.tensor(num_of_sequence).sum()
    avg_num_of_agents_in_each_sequence = torch.abs(torch.FloatTensor(num_of_agents_in_each_sequence)).mean()
    avg_velocity_x_of_agents_in_each_sequence = torch.abs(torch.tensor(velocity_x_of_agents_in_each_sequence)).mean()
    avg_velocity_y_of_agents_in_each_sequence = torch.abs(torch.tensor(velocity_y_of_agents_in_each_sequence)).mean()
    avg_acc_x_of_agents_in_each_sequence = torch.abs(torch.tensor(acc_x_of_agents_in_each_sequence)).mean()
    avg_acc_y_of_agents_in_each_sequence = torch.abs(torch.tensor(acc_y_of_agents_in_each_sequence)).mean()

    std_num_of_agents_in_each_sequence = torch.abs(torch.FloatTensor(num_of_agents_in_each_sequence)).std()
    std_velocity_x_of_agents_in_each_sequence = torch.abs(torch.tensor(velocity_x_of_agents_in_each_sequence)).std()
    std_velocity_y_of_agents_in_each_sequence = torch.abs(torch.tensor(velocity_y_of_agents_in_each_sequence)).std()
    std_acc_x_of_agents_in_each_sequence = torch.abs(torch.tensor(acc_x_of_agents_in_each_sequence)).std()
    std_acc_y_of_agents_in_each_sequence = torch.abs(torch.tensor(acc_y_of_agents_in_each_sequence)).std()

    print(f'--------dataset summary-----------')
    print(f'num_of_sequence: {sum_num_of_sequence}')
    print(f'avg_num_of_agents_in_each_sequence: {avg_num_of_agents_in_each_sequence}')
    print(f'avg_velocity_x_of_agents_in_each_sequence: {avg_velocity_x_of_agents_in_each_sequence}')
    print(f'avg_velocity_y_of_agents_in_each_sequence: {avg_velocity_y_of_agents_in_each_sequence}')
    print(f'avg_acc_x_of_agents_in_each_sequence: {avg_acc_x_of_agents_in_each_sequence}')
    print(f'avg_acc_y_of_agents_in_each_sequence: {avg_acc_y_of_agents_in_each_sequence}')
    print('----------STD--------------')
    print(f'std_num_of_agents_in_each_sequence: {std_num_of_agents_in_each_sequence}')
    print(f'std_velocity_x_of_agents_in_each_sequence: {std_velocity_x_of_agents_in_each_sequence}')
    print(f'std_velocity_y_of_agents_in_each_sequence: {std_velocity_y_of_agents_in_each_sequence}')
    print(f'std_acc_x_of_agents_in_each_sequence: {std_acc_x_of_agents_in_each_sequence}')
    print(f'std_acc_y_of_agents_in_each_sequence: {std_acc_y_of_agents_in_each_sequence}')
    return True


def analyze_dataset_split_train_val_test(train_loaders, val_loaders, test_loaders):
    dict_loaders = {'train': train_loaders, 'val': val_loaders, 'test': test_loaders}
    for phase in dict_loaders.keys():
        loaders = dict_loaders[phase]

        list_num_of_sequence = []
        list_avg_num_of_agents_in_each_sequence = []
        list_avg_velocity_x_of_agents_in_each_sequence = []
        list_avg_velocity_y_of_agents_in_each_sequence = []
        list_avg_acc_x_of_agents_in_each_sequence = []
        list_avg_acc_y_of_agents_in_each_sequence = []
        for loader in loaders.values():
            num_of_agents_in_each_sequence = []
            velocity_x_of_agents_in_each_sequence = []
            velocity_y_of_agents_in_each_sequence = []
            acc_x_of_agents_in_each_sequence = []
            acc_y_of_agents_in_each_sequence = []

            num_seq = loader.dataset.num_seq
            vel_traj = loader.dataset.obs_traj_rel
            seq_start_end = loader.dataset.seq_start_end
            for index in seq_start_end:
                start, end = index
                vel_sequence = vel_traj[start:end, :, 1:]
                acc_sequence = vel_sequence[:, :, 1:] - vel_sequence[:, :, :-1]
                num_of_agents_in_each_sequence.append(end-start)
                velocity_x_of_agents_in_each_sequence.append(torch.abs(vel_sequence[:, 0, :]).mean())
                velocity_y_of_agents_in_each_sequence.append(torch.abs(vel_sequence[:, 1, :]).mean())
                acc_x_of_agents_in_each_sequence.append(torch.abs(acc_sequence[:, 0, :]).mean())
                acc_y_of_agents_in_each_sequence.append(torch.abs(acc_sequence[:, 1, :]).mean())

            list_num_of_sequence.append(num_seq)
            list_avg_num_of_agents_in_each_sequence.append(torch.abs(torch.FloatTensor(num_of_agents_in_each_sequence)).mean())
            list_avg_velocity_x_of_agents_in_each_sequence.append(torch.abs(torch.tensor(velocity_x_of_agents_in_each_sequence)).mean())
            list_avg_velocity_y_of_agents_in_each_sequence.append(torch.abs(torch.tensor(velocity_y_of_agents_in_each_sequence)).mean())
            list_avg_acc_x_of_agents_in_each_sequence.append(torch.abs(torch.tensor(acc_x_of_agents_in_each_sequence)).mean())
            list_avg_acc_y_of_agents_in_each_sequence.append(torch.abs(torch.tensor(acc_y_of_agents_in_each_sequence)).mean())

        num_of_sequence = torch.tensor(list_num_of_sequence).sum()
        avg_num_of_agents_in_each_sequence = torch.abs(torch.tensor(list_avg_num_of_agents_in_each_sequence)).mean()
        avg_velocity_x_of_agents_in_each_sequence = torch.abs(torch.tensor(list_avg_velocity_x_of_agents_in_each_sequence)).mean()
        avg_velocity_y_of_agents_in_each_sequence = torch.abs(torch.tensor(list_avg_velocity_y_of_agents_in_each_sequence)).mean()
        avg_acc_x_of_agents_in_each_sequence = torch.abs(torch.tensor(list_avg_acc_x_of_agents_in_each_sequence)).mean()
        avg_acc_y_of_agents_in_each_sequence = torch.abs(torch.tensor(list_avg_acc_y_of_agents_in_each_sequence)).mean()
        std_num_of_agents_in_each_sequence = torch.abs(torch.tensor(list_avg_num_of_agents_in_each_sequence)).std()
        std_velocity_x_of_agents_in_each_sequence = torch.abs(torch.tensor(list_avg_velocity_x_of_agents_in_each_sequence)).std()
        std_velocity_y_of_agents_in_each_sequence = torch.abs(torch.tensor(list_avg_velocity_y_of_agents_in_each_sequence)).std()
        std_acc_x_of_agents_in_each_sequence = torch.abs(torch.tensor(list_avg_acc_x_of_agents_in_each_sequence)).std()
        std_acc_y_of_agents_in_each_sequence = torch.abs(torch.tensor(list_avg_acc_y_of_agents_in_each_sequence)).std()

        print(f'--------{phase}-----------')
        print(f'num_of_sequence: {num_of_sequence}')
        print(f'avg_num_of_agents_in_each_sequence: {avg_num_of_agents_in_each_sequence}')
        print(f'avg_velocity_x_of_agents_in_each_sequence: {avg_velocity_x_of_agents_in_each_sequence}')
        print(f'avg_velocity_y_of_agents_in_each_sequence: {avg_velocity_y_of_agents_in_each_sequence}')
        print(f'avg_acc_x_of_agents_in_each_sequence: {avg_acc_x_of_agents_in_each_sequence}')
        print(f'avg_acc_y_of_agents_in_each_sequence: {avg_acc_y_of_agents_in_each_sequence}')
        print('----------STD--------------')
        print(f'std_num_of_agents_in_each_sequence: {std_num_of_agents_in_each_sequence}')
        print(f'std_velocity_x_of_agents_in_each_sequence: {std_velocity_x_of_agents_in_each_sequence}')
        print(f'std_velocity_y_of_agents_in_each_sequence: {std_velocity_y_of_agents_in_each_sequence}')
        print(f'std_acc_x_of_agents_in_each_sequence: {std_acc_x_of_agents_in_each_sequence}')
        print(f'std_acc_y_of_agents_in_each_sequence: {std_acc_y_of_agents_in_each_sequence}')
    return True


class TrajectoryDataset(torch.utils.data.Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, split_data_path, preprocessed_data_path, dataset_name, phase,
            obs_len=8, pred_len=12, skip=1, threshold=0.002, min_ped=1,
            process_flag='frame', rotate_flag='False', reverse_flag='False'
    ):
        """
        Args:
        - data_path, dataset_name: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        - process_flag: frame or agent
        - rotate_flag: true or false when phase is training
        - reverse_flag: true or false when phase is training
        """
        super(TrajectoryDataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len

        self.process_flag = process_flag
        self.reverse_flag = reverse_flag
        self.rotate_flag = rotate_flag

        load_name = os.path.join(preprocessed_data_path, dataset_name.split('.')[0] +
                                 f'_obs_{obs_len}_pred_{pred_len}_process_{process_flag}_{phase}.pickle')

        if os.path.exists(load_name):
            with open(load_name, 'rb') as f:
                data = pickle.load(f)
            print('loaded preprocessed data file from {}'.format(load_name))
        else:
            os.makedirs(preprocessed_data_path, exist_ok=True)
            data = self.create_pikle(split_data_path, dataset_name, load_name, obs_len, pred_len, skip, threshold, min_ped)

        if len(data) > 0:
            # frame: each element of loss_mask_list is a [num_peds_considered, num_peds_considered] identity matrix
            # agent: each element of loss_mask_list is a [num_peds_considered, seq_len] = 1
            # frame, agent下的loss_mask不一致，但会在seq_collate阶段统一loss_mask的表示，故不会造成任何影响
            self.num_seq, self.traj, self.obs_traj, self.pred_traj, self.obs_traj_rel, self.pred_traj_rel, \
            self.loss_mask, self.seq_start_end, self.non_linear_ped = data
            # self.num_seq: int
            # self.traj: [_, 2, seq_len], seq_len = obs_len + pred_len
            # obs_traj: [_, 2, obs_len]
            # pred_traj: [_, 2, pred_len]
            # obs_traj_rel: [_, 2, obs_len], where obs_traj_rel[:, :, 0] = 0, obs_traj_rel[:, :, 1:] = obs_traj[:, :, 1:] - obs_traj[:, :, :1]
            # self.loss_mask: [num_seq, num_peds_considered, num_peds_considered/seq_len]
            # self.seq_start_end: can be computed from self.loss_mask

            if (phase == 'train') and (self.reverse_flag == 'True'):
                traj = self.traj.clone()
                for t in traj:
                    t = np.array(t)
                    reverse_t = np.flip(t, axis=1).copy()
                    self.traj = torch.cat([self.traj, torch.from_numpy(reverse_t).reshape(1, 2, -1)], dim=0)
                self.loss_mask = np.tile(self.loss_mask, (2, ))
                self.non_linear_ped = np.tile(self.non_linear_ped, (2,))
                self.non_linear_ped = torch.from_numpy(self.non_linear_ped).type(torch.float)
                self.seq_start_end = self.seq_start_end * 2
                self.num_seq = self.num_seq * 2

            # Error: only rotate in self.traj and self.loss_mask, while there are other elements of data
            if (phase == 'train') and (self.rotate_flag == 'True'):
                rotation_matrix = lambda alpha: np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
                angles = [0.25 * np.pi, 0.5 * np.pi, -0.25 * np.pi, -0.5 * np.pi]
                traj = self.traj.clone().permute(0, 2, 1).reshape(-1, 2)
                for angle in angles:
                    self.traj = torch.cat([
                        self.traj, np.matmul(traj, rotation_matrix(angle)).reshape(-1, obs_len+pred_len, 2).permute(0, 2, 1)],
                        dim=0)
                self.loss_mask = np.tile(self.loss_mask, (len(angles), ))
                self.non_linear_ped = np.tile(self.non_linear_ped, (len(angles),))
                self.non_linear_ped = torch.from_numpy(self.non_linear_ped).type(torch.float)
                self.seq_start_end = self.seq_start_end * len(angles)
                self.num_seq = self.num_seq * len(angles)
        else:
            self.num_seq = 0

    def find_min_time(self, t1, t2):
        """given two time frame arrays, find then min dist between starts"""
        min_d = 999999999
        for t in t2:
            if abs(t1[0] - t) < min_d:
                min_d = abs(t1[0] - t)

        for t in t1:
            if abs(t2[0] - t) < min_d:
                min_d = abs(t2[0] - t)

        return min_d

    def find_min_dist(self, p1x, p1y, p2x, p2y):
        '''given two time frame arrays, find then min dist between starts'''
        min_d = 999999999
        for i in range(len(p1x)):
            for j in range(len(p1x)):
                if ((p2x[i] - p1x[j]) ** 2 + (p2y[i] - p1y[j]) ** 2) ** 0.5 < min_d:
                    min_d = ((p2x[i] - p1x[j]) ** 2 + (p2y[i] - p1y[j]) ** 2) ** 0.5

        return min_d

    def social_and_temporal_filter(self, p1_key, p2_key, all_data_dict, time_thresh=10, dist_tresh=50):
        p1_traj, p2_traj = np.array(all_data_dict[p1_key]), np.array(all_data_dict[p2_key])
        p1_time, p2_time = p1_traj[:, 0], p2_traj[:, 0]
        p1_x, p2_x = p1_traj[:, 2], p2_traj[:, 2]
        p1_y, p2_y = p1_traj[:, 3], p2_traj[:, 3]

        # if they are the same person id, no self loops
        if all_data_dict[p1_key][0][1] == all_data_dict[p2_key][0][1]:
            return False
        if self.find_min_time(p1_time, p2_time) > time_thresh:
            return False
        # if self.find_min_dist(p1_x, p1_y, p2_x, p2_y) > dist_tresh:
        #     return False

        return True

    def poly_fit(self, traj, traj_len, threshold):
        """
        Input:
        - traj: Numpy array of shape (2, traj_len)
        - traj_len: Len of trajectory
        - threshold: Minimum error to be considered for non linear traj
        Output:
        - int: 1 -> Non Linear 0-> Linear
        """
        t = np.linspace(0, traj_len - 1, traj_len)
        res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
        res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
        if res_x + res_y >= threshold:
            return 1.0
        else:
            return 0.0

    def create_pikle(self, split_data_path, dataset_name, save_name, obs_len, pred_len, skip, threshold, min_ped):

        print("Start pickling. ")

        all_files = [os.path.join(split_data_path, dataset_name)]
        num_peds_in_seq = []  # number of peds considered in the sequence
        seq_list = []    # each is a [num_peds_considered, 2, seq_len], here 2 means x, y respectively
        seq_list_rel = []
        # frame: each element of loss_mask_list is a [num_peds_considered, num_peds_considered] identity matrix
        # agent: each element of loss_mask_list is a [num_peds_considered, seq_len] = 1
        # frame, agent下的loss_mask不一致，但会在seq_collate阶段统一loss_mask的表示，故不会造成任何影响
        loss_mask_list = []
        non_linear_ped = []   # number of non linear trajectory considerded in the sequence
        seq_len = obs_len + pred_len
        for path in all_files:
            # distinguish scene
            if path[-3:] == 'csv':
                data = pd.read_csv(path)
            else:
                data = []
                with open(os.path.join(path), 'r') as f:
                    for line in f:
                        line = line.strip().split('\t')
                        line = [float(i) for i in line if i != '']
                        # line = [frame agentID pos_x pos_z pos_y v_x v_z v_y]
                        data.append([int(float(line[0])), int(float(line[1])), float(line[2]), float(line[3])])
            # data = [frame, agentID, x, y]
            data = np.asarray(data)

            if self.process_flag == 'frame':
                '''
                there are 3 preprocessing steps which are needed to consider whether they are reasonable. 
                1. rel_cur_ped_seq[0] = 0 
                   (what if rel_cur_ped_seq[0] is quite different from rel_cur_ped_seq[1])
                2. non_linear
                   (how can we ues non_linear, and how to compute non_linear)
                3. num_peds_consider > min_peds (where min_peds = 1)
                   (why can not predict a sequence where there are only 1 ped)
                '''
                frames = sorted(np.unique(data[:, 0]))
                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])
                num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))

                for idx in range(0, num_sequences * skip + 1, skip):
                    curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len))
                    curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, seq_len))
                    curr_loss_mask = np.zeros((len(peds_in_curr_seq), len(peds_in_curr_seq)))
                    num_peds_considered = 0
                    _non_linear_ped = []
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        # pad_end - pad_front == seq_len (= obs_len + pred_len)
                        # (to ensure it is a continuous sequence, not intermittent, e.g., 1, 3, 4, 7, 12)
                        if pad_end - pad_front != seq_len:
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                        curr_ped_seq = curr_ped_seq
                        # Make coordinates relative
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                        _idx = num_peds_considered
                        if curr_ped_seq.shape[1] != seq_len:
                            print(f'ped id {ped_id} has no seq_len trajs from frame {curr_ped_seq[0, 0]} to {curr_ped_seq[-1, 0]}')
                            continue
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                        # Linear vs Non-Linear Trajectory
                        _non_linear_ped.append(self.poly_fit(curr_ped_seq, pred_len, threshold))
                        curr_loss_mask[_idx, _idx] = 1
                        num_peds_considered += 1

                    if num_peds_considered > min_ped:
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_peds_considered)
                        # curr_loss_mask[:num_peds_considered, :num_peds_considered] is an identity matrix,
                        # i.e., (i, i)=1, (i, j) = 0
                        loss_mask_list.append(curr_loss_mask[:num_peds_considered, :num_peds_considered])
                        seq_list.append(curr_seq[:num_peds_considered])
                        seq_list_rel.append(curr_seq_rel[:num_peds_considered])
            elif self.process_flag == 'agent':
                '''
                there are 5 preprocessing steps which are needed to consider whether they are reasonable. 
                1. data_by_agent[counter] = agent[i: i+ seq_len, :]  (solved by qtw)
                   (do not consider whether the time is cons)
                   (do not sort by time)
                2. cur_seq_rel[0] = 0 
                   (what if rel_cur_ped_seq[0] is quite different from rel_cur_ped_seq[1])
                3. non_linear
                   (how can we ues non_linear, and how to compute non_linear)
                4. find_min_time() use abs are needed to consider whether they are reasonable. 
                   (only during obs_len timesteps, neighbors are observed)
                5. find_min_time > time_thresh (where time_thresh = obs_len) 
                   (whether > or >=)
                '''
                frames = sorted(np.unique(data[:, 0]))
                agents = sorted(np.unique(data[:, 1]))
                data_by_agent = {}
                counter = 0
                for agent in agents:
                    agent_data = data[agent == data[:, 1]]
                    agent_data = agent_data[agent_data[:, 0].argsort()]    # sort agent_data by time
                    len_agent_data = len(agent_data)
                    if len_agent_data >= seq_len:
                        for i in range(0, len_agent_data-seq_len+1):
                            if (frames.index(agent_data[i+seq_len-1, 0]) - frames.index(agent_data[i, 0]) + 1) != seq_len:
                                continue
                            data_by_agent[counter] = agent_data[i:i+seq_len, :]
                            counter += 1
                all_data_dict = data_by_agent.copy()

                while len(list(data_by_agent.keys())) > 0:
                    curr_keys = list(data_by_agent.keys())
                    init_agent_len = len(curr_keys)

                    counter = 0
                    curr_seq = np.zeros((init_agent_len, 2, seq_len))
                    curr_seq_rel = np.zeros((init_agent_len, 2, seq_len))
                    curr_loss_mask = np.zeros((init_agent_len, seq_len))
                    _non_linear_ped = []

                    curr_seq[counter, :, :] = np.transpose(all_data_dict[curr_keys[0]][:, 2:])
                    curr_seq_rel[counter, :, 1:] = curr_seq[counter, :, 1:] - curr_seq[counter, :, :-1]
                    _non_linear_ped.append(self.poly_fit(curr_seq[counter, :, :], pred_len, threshold))
                    curr_loss_mask[counter] = 1
                    counter += 1
                    del data_by_agent[curr_keys[0]]

                    for i in range(1, len(curr_keys)):
                        if self.social_and_temporal_filter(curr_keys[0], curr_keys[i], all_data_dict, time_thresh=obs_len):
                            curr_seq[counter, :, :] = np.transpose(all_data_dict[curr_keys[i]][:, 2:])
                            curr_seq_rel[counter, :, 1:] = curr_seq[counter, :, 1:] - curr_seq[counter, :, :-1]
                            _non_linear_ped.append(self.poly_fit(curr_seq[counter, :, :], pred_len, threshold))
                            curr_loss_mask[counter] = 1
                            counter += 1
                            del data_by_agent[curr_keys[i]]

                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(counter)
                    loss_mask_list.append(curr_loss_mask[:counter])
                    seq_list.append(curr_seq[:counter])
                    seq_list_rel.append(curr_seq_rel[:counter])
            else:
                raise ValueError(f'wrong process flag {self.process_flag}')

        num_seq = len(seq_list)
        if num_seq == 0:
            print(f'this file can not generate dataloader: {dataset_name}')
            return []

        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        # loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        traj = torch.from_numpy(seq_list[:, :, :]).type(torch.float)
        obs_traj = torch.from_numpy(
            seq_list[:, :, :obs_len]).type(torch.float)
        pred_traj = torch.from_numpy(
            seq_list[:, :, obs_len:]).type(torch.float)
        obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :obs_len]).type(torch.float)
        pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, obs_len:]).type(torch.float)
        # loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        loss_mask = loss_mask_list
        non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        all_data = [num_seq, traj, obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, loss_mask, seq_start_end, non_linear_ped]

        with open(save_name, 'wb') as f:
            pickle.dump(all_data, f)

        print(f"Finish Pickling and Save Pickle file. ")

        return all_data

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.traj[start:end, :],
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[index]
        ]
        return out


if __name__ == '__main__':
    # change_file_into_csv()
    # compare_txt_csv_file()
    # train_fraction = 0.8
    # val_fraction = 0.1
    # split_train_val_test_data(train_fraction, val_fraction)
    print('finish')
