# TODO Mark the fastest and slowest with different colres (text or bar)
# TODO If different kernels then bars together
import sys
import os
import shutil
import configparser
import json
import copy

from os import listdir
from os.path import isfile, isdir, join, exists, basename, dirname, splitext 

from pathlib import Path

from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt

def autolabel(ax, rects, xpos='center', ypos=0, ypos_limit=0, rot=0):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        text = '{0:0.2f}'.format(height) if isinstance(height, float) else '{}'.format(height)
        if ypos_limit < height:
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], height - ypos*height, text, ha=ha[xpos], va='bottom', rotation=rot)
        else :
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.05*height, text, ha=ha[xpos], va='bottom', rotation=rot)

from argparse import ArgumentParser

parser = ArgumentParser(description='Plot barcharts for execution results.')
parser.add_argument('-i', '--input_path', dest='input_path', default='')
parser.add_argument('-o', '--output_path', dest='output_path', default='')
parser.add_argument('--plot_title', dest='plot_title', default='Kernel comparison')
parser.add_argument('--output_filename', dest='output_filename', default='memsize')
parser.add_argument('--height', type=float, dest='height', default=15)
parser.add_argument('--width', type=float, dest='width', default=15)
parser.add_argument('-d', '--dataset', type=int, dest='dataset', default=0)
parser.add_argument('--datasets', dest='datasets', nargs='+', default=' ')
# parser.add_argument('-k', '--kernel_type', dest='kernel_type', default=' ')
parser.add_argument('-k','--kernel_types', dest='kernel_types', nargs='+', default=' ')
parser.add_argument('-p', '--plots', dest='plots', nargs='+', default=[0, 1])

args = parser.parse_args()
args.plots = [int(i) for i in args.plots]

config = configparser.ConfigParser()   
config_file_path = r'C:\Work\SimCorp\stl-hpc-research\scripts\research_barcharts_diku.config'
config.read(config_file_path)

# read in results path (from program arguments or config file)
input_path = ''
if args.input_path:
    # print(sys.argv[1:])
    input_path = args.input_path
else:
    input_path = config['Paths']['results']

output_path = ''
if args.output_path:
    # print(sys.argv[1:])
    output_path = args.output_path
else:
    output_path = config['Paths']['results']

cpu_results_path = config['Paths']['cpu_results']
device_results_filenames = json.loads(config['Paths']['device_results'])

# get data
# print(input_path)
if not os.path.isfile(input_path):
    exit

limited_datasets_path = json.loads(config['Paths']['limited_datasets'])

limited_datasets = []

input_path_dir = dirname(input_path)
output_path_dir = dirname(output_path)

delimeter = ','

for path in limited_datasets_path:
    if isdir(path):
        filenames = [splitext(basename(f))[0] for f in listdir(path) if isfile(join(path, f)) and not 'yield.in' in f]
        limited_datasets += filenames

if len(limited_datasets) == 0:
    limited_datasets = limited_datasets_path

precision_dict = {'float': 'Single', 'double': 'Double'}
tech_dict = {'principiac': 'SCD Prod', 'seqc': 'Sequential C', 'openmp': 'OpenMP', 'cuda': 'CUDA', 'opencl': 'OpenCL', 'futhark-c': 'Futhark-C', 'futhark-opencl': 'Futhark-OpenCL'}
platform_dict = {'intel': 'CPU', 'p100': 'P100', 'gtx780': 'GTX780'}
device_dict = {'p100': 'P100', 'gtx780': 'GTX780'}
version_dict = {'cuda-option': 'Opt/T', 'cuda-multi': 'Opts/TB'}
kernel_dict = {1: 'NOOPT', 2: 'PAD_GLOBAL', 3: 'PAD_TB', 4: 'PAD_WARP'}
sort_dict = {'-': 'No sort', 'H': 'Height desc', 'h': 'Height asc', 'W': 'Width desc', 'w': 'Width asc'}
optimization_dict = {'': 'All optimizations', 'PAD_GLOBAL': 'w/o TB padding', 'No sort': 'w/o reordering'}

# datasets = set()
datasets = limited_datasets if limited_datasets else set()
results_dict = dict()

for filename in device_results_filenames:
    with open(input_path + filename) as f:
        lines = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        # lines = [x.strip().split(' ') for x in [ line for line in lines if '2018-'in line and not 'Expected' in line and not 'getLastCudaError()' in line]]
        lines = [x.strip().split(delimeter) for x in [ line for line in lines if not 'file' in line]]

        (device, version) = splitext(basename(filename))[0].split('_')

        kernels = set()
        # get kernels existing in results file
        for line in lines:
            kernelVersion = 0
            try:
                kernelVersion = int(line[3])
            except ValueError:
                continue
            key_str = version_dict.get(version) + ' ' + kernel_dict.get(kernelVersion) + ' ' + sort_dict.get(line[5]) + ' ' + line[4] + ' ' + precision_dict.get(line[1]) + ' ' + device_dict.get(device)
            kernels.add((key_str, kernelVersion, line[5], int(line[4]), line[1], device, version))
            if not len(limited_datasets) > 0: # and exists(line[10]):
                dataset = splitext(basename(line[10]))[0]
                datasets.add(dataset)

        kernels_list = list(kernels)
        kernels_list.sort(key=itemgetter(4,1,2,3))

        datasets = sorted(datasets)
        datasets_count = len(datasets)
        memsizes = [[]] * datasets_count
        datesets_iter = iter(datasets)

        # assign result per dataset to each of kernels
        results_dict.update(dict((kernel[0], memsizes.copy()) for kernel in kernels))
        for line in lines:
            for key in kernels_list:
                if key[1] == int(line[3]) and key[2] == line[5] and key[3] == int(line[4]) and key[4] == line[1] and key[5] == device and key[6] == version: # kernel, sort, blocksize, precision match
                    # print(version + ' ' + device + ' ' + str(line) + ' ' + key[0])
                    memsize = 0
                    try:
                        memsize = int(line[8]) # memsize
                    except ValueError:
                        memsize = 0
                    # if exists(line[10]):
                    # dataset = splitext(basename(line[10]))[0]
                    dataset = line[0]
                    index = 0
                    for index_dataset in datasets:
                        if dataset == index_dataset:
                            if len(results_dict[key[0]][index]) == 0:
                                results_dict[key[0]][index] = [memsize]
                            else:
                                results_dict[key[0]][index].append(memsize)
                            break
                        else:
                            index += 1
                    break

# average results from many trials
# for kernel, memsizes in results_dict.items():
    # for dataset_index in range(datasets_count):
        # memsizes[dataset_index] = int(np.mean(memsizes[dataset_index]))

# print and plot the best results and speedups in comparison to OpenMP CPU code for each dataset
colors = ['SkyBlue', 'IndianRed', 'Green', 'Orange']
offset = [-1.5, -0.5, 0.5, 1.5]
width = 0.225  # the width of the bars

if 0 in args.plots:
    for precision in precision_dict.values():
        memsize_fig, memsize_ax = plt.subplots()
        memsize_rects = []

        inds = np.arange(datasets_count)  # the x locations for the groups

        memsize_ax.set_ylabel('Memsize [byte]')
        memsize_ax.set_title('Comparison of the memory size used across datasets, devices and versions (' + precision + ' precision)')
        memsize_ax.set_xticks(inds)
        memsize_ax.set_xticklabels(datasets)
        # for tick in memsize_ax.get_xticklabels():
        #     tick.set_rotation(45)

        count = 0
        for device in device_dict.values():
            for version in version_dict.values():
                filtered_results_dict = {k:v for (k,v) in results_dict.items() if precision in k and version in k and device in k}
                filtered_kernel_names = np.array(list(filtered_results_dict.keys())).tolist()
                filtered_best_results = []
                
                for dataset_index in range(datasets_count):
                    filtered_kernel_memsizes = np.array(list(filtered_results_dict.values()))[:,dataset_index].tolist()
                    best_result = round(min(filtered_kernel_memsizes)[0]/(1024*1024),3)
                    filtered_best_results.append(best_result)
                    best_result_index = filtered_kernel_memsizes.index(min(filtered_kernel_memsizes))

                    print(datasets[dataset_index] + ' ' +
                        str(best_result) + ' MB ' +
                        filtered_kernel_names[best_result_index])

                # print(filtered_best_results)
                truncated_results = ["{0:0.3f}".format(s) for s in filtered_best_results]
                results_str = ' '.join(truncated_results)
                print(results_str)

                label = device + " " + version
                
                kernel_rects = memsize_ax.bar(inds + offset[count]*width, filtered_best_results, width, color=colors[count], label=label)
                memsize_rects.append(kernel_rects)

                count = count + 1

        rect_heights = []
        for kernel_rects in memsize_rects:
            for rect in kernel_rects:
                rect_heights.append(rect.get_height())
        max_rect_height = max(rect_heights)
        
        for kernel_rects in memsize_rects: 
            autolabel(memsize_ax, kernel_rects, 'center', 0.3, 0.9*max_rect_height, 90)
            
        memsize_ax.legend()
        # plt.show()
        memsize_fig.set_size_inches(args.height, args.width)
        output_path = output_path_dir + '\\figures\\' + 'memsize_best_' + precision + '.eps'
        memsize_fig.savefig(output_path, format='eps', dpi=1200, bbox_inches='tight')

offset = [-0.5, 0.5]
width = 0.45  # the width of the bars

if 1 in args.plots:
    for precision in precision_dict.values():
        for device in device_dict.values():
            for version in version_dict.values():
                memsize_fig, memsize_ax = plt.subplots()
                memsize_rects = []

                inds = np.arange(datasets_count)  # the x locations for the groups

                memsize_ax.set_ylabel('Memsize [MB]')
                memsize_ax.set_title('Comparison of optimization impact on memory size across datasets (' + device + ', ' + version + ', ' + precision + ' precision)')
                memsize_ax.set_xticks(inds)
                memsize_ax.set_xticklabels(datasets)
                # for tick in memsize_ax.get_xticklabels():
                #     tick.set_rotation(45)

                count = 0
                filtered_best_results = []
                for optimization in optimization_dict.keys():
                    filtered_results_dict = {k:v for (k,v) in results_dict.items() if precision in k and version in k and device in k and optimization in k}
                    filtered_kernel_names = np.array(list(filtered_results_dict.keys())).tolist()
                    filtered_best_names = []
                    optimization_best_results = []

                    for dataset_index in range(datasets_count):
                        filtered_kernel_memsizes = np.array(list(filtered_results_dict.values()))[:,dataset_index].tolist()
                        best_result = round(min(filtered_kernel_memsizes)[0]/(1024*1024),3)
                        optimization_best_results.append(best_result)
                        best_result_index = filtered_kernel_memsizes.index(min(filtered_kernel_memsizes))
                        name = filtered_kernel_names[best_result_index]
                        filtered_best_names.append(name)
                        print(datasets[dataset_index] + ' ' +
                            str(best_result) + ' MB ' +
                            name)

                    # print(optimization_best_results)
                    truncated_results = ["{0:0.3f}".format(s) for s in optimization_best_results]
                    results_str = ' '.join(truncated_results)
                    print(results_str)

                    if optimization == '':
                        filtered_best_results = optimization_best_results.copy()
                    else:
                        results = [a/b for a,b in zip(optimization_best_results,filtered_best_results)]
                        
                        kernels_length = len(filtered_results_dict)

                        label = optimization_dict[optimization]
                        
                        kernel_rects = memsize_ax.bar(inds + offset[count]*width, results, width, color=colors[count], label=label)
                        memsize_rects.append(kernel_rects)

                        count = count + 1

                rect_heights = []
                for kernel_rects in memsize_rects:
                    for rect in kernel_rects:
                        rect_heights.append(rect.get_height())
                max_rect_height = max(rect_heights)
                
                for kernel_rects in memsize_rects: 
                    autolabel(memsize_ax, kernel_rects, 'center', 0.2, 0.9*max_rect_height, 45)

                memsize_ax.legend()
                # plt.show()
                memsize_fig.set_size_inches(args.height, args.width)
                version_key = list(version_dict.keys())[list(version_dict.values()).index(version)]

                output_path = output_path_dir + '\\figures\\' + 'memsize_opt_impact_' + precision + '_' + device + '_' + version_key + '.eps'
                memsize_fig.savefig(output_path, format='eps', dpi=1200, bbox_inches='tight')

# for key, value in results_dict.items():
#     value.sort(key=itemgetter(0))
if 2 in args.plots:
    for dataset_index in range(datasets_count):
        if (dataset_index in args.datasets):
            fig, ax = plt.subplots()
            width = 0.9  # the width of the bars
            rects = []

            for kernel_type in args.kernel_types:

                filtered_results_dict = {k:v for (k,v) in results_dict.items() if kernel_type in k}

                kernels_length = len(filtered_results_dict)
                ind = np.arange(kernels_length)  # the x locations for the groups

                kernel_names = filtered_results_dict.keys()
                # kernel_names = results_dict.keys()
                # index = -len(filtered_results_dict)/2
                kernel_memsizes = np.array(list(filtered_results_dict.values()))[:,dataset_index].tolist()
                kernel_memsizes = [item for sublist in kernel_memsizes for item in sublist]
                kernel_rects = ax.bar(ind, kernel_memsizes, width, label=kernel_names)
                rects.append(kernel_rects)

            min_memsize = min(float(s) for s in (t for t in kernel_memsizes if t > 0))
            max_memsize = max(float(s) for s in (t for t in kernel_memsizes if t > 0))

        # ylim_min = min_memsize
        # ylim_max = max_memsize
        # ax.set_ylim(bottom=ylim_min)
            # ylim_max = 80000
            # ax.set_ylim([0,ylim_max])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Execution Time [ms]')
            ax.set_title(args.plot_title + '\nDataset: ' + datasets[dataset_index])
            ax.set_xticks(ind)
            ax.set_xticklabels(kernel_names)
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
                # tick.set_rotation('vertical')
            # ax.legend(loc='center right', bbox_to_anchor=(1,0.5), fontsize='small')
            # ax.legend(bbox_to_anchor=(1, 0.5), loc=2, borderaxespad=0.)
            # ax.legend(bbox_to_anchor=(0., 1.5, 1., 2.102), loc=8, ncol=5, mode='expand', borderaxespad=0.)

            for i, v in enumerate(kernel_memsizes):
            # if v == 0:
            #     ax.text(ylim_min + 5, i - 0.1, 'N/A')
            # else:
                    ax.text(v + 5, i - 0.2, str(v))

            for kernel_rects in rects:
                autolabel(kernel_rects, 'center', 0.1)

            # plt.show()
            fig.set_size_inches(args.height, args.width)
            output_path = output_path_dir + '\\figures\\'  + args.output_filename +'_' + datasets[dataset_index] + '.eps'
            fig.savefig(output_path, format='eps', dpi=1200, bbox_inches='tight')
