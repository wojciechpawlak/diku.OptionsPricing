# TODO Mark the fastest and slowest with different colres (text or bar)
# TODO If different kernels then bars together
import sys
import os
import shutil
import configparser
import json
import copy
import math

from os import listdir
from os.path import isfile, isdir, join, exists, basename, dirname, splitext 

from pathlib import Path

from operator import itemgetter

from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

import collections

# Constants
delimeter = ','

precision_dict = {'float': 'Single', 'double': 'Double'}
tech_dict = {'principiac': 'SCD Prod', 'seqc': 'Sequential C', 'openmp': 'OpenMP', 'cuda': 'CUDA', 'opencl': 'OpenCL', 'futhark-c': 'Futhark-C', 'futhark-opencl': 'Futhark-OpenCL'}
platform_dict = {'intel': 'CPU', 'v100': 'V100', 'p100': 'P100', 'gtx780': 'GTX780'}
cpu_dict = {'52core': '52core', '32core': '32core'}
# device_dict = {'v100': 'V100', 'p100': 'P100', 'gtx780': 'GTX780'}
device_dict = {'v100': 'V100', 'gtx780': 'GTX780'}
# version_dict = {'cuda-option': 'gpu-outer', 'cuda-multi': 'gpu-flat', 'cpu': 'cpu-mt+vect', 'ql': 'quantlib-par'}
version_dict = {'cuda-option': 'gpu-outer', 'cuda-multi': 'gpu-flat', 'cpu': 'cpu-mt+vect'}
kernel_dict = {1: 'NOOPT', 2: 'PAD_GLOBAL', 3: 'PAD_TB', 4: 'PAD_WARP'}
sort_dict = {'-': 'No sort', 'H': 'Height desc', 'h': 'Height asc', 'W': 'Width desc', 'w': 'Width asc'}
optimization_dict = {'': 'All optimizations', 'NOOPT': 'w/o coalescing', 'PAD_GLOBAL': 'w/o TB padding', 'No sort': 'w/o reordering'}
dataset_dict = {'0_UNIFORM_3000': 'U1', '0_UNIFORM_100000': 'U2', '1_RAND_100000': 'R1', '1_RAND_NORMH_100000': 'R2', '1_RAND_NORMW_100000': 'R3', '4_SKEWED_1_100000': 'S1', '4_SKEWED_INV_1_100000': 'S2'}

# FLOPs
# futhark flops-memops.fut script
# "0_UNIFORM_10000" "0_UNIFORM_100000" "1_RAND_100000" "4_SKEWED_100000"
# flops_OptT_single = [1172990000,11729900000,335986186408,33987585746]
# flops_OptsTB_single = [1172990000,11729900000,335986186408,33987585746]
# flops_OptT_double = [1172990000,11729900000,335986186408,33987585746]
# flops_OptsTB_double = [1172990000,11729900000,335986186408,33987585746]

# flops_OptT_single = [335986186408,33987585746]
# flops_OptsTB_single = [335986186408,33987585746]
# flops_OptT_double = [335986186408,33987585746]
# flops_OptsTB_double = [335986186408,33987585746]

# NSight Profiler
# flops_OptT_double = [264422436545,22444351687]
# flops_OptsTB_double = [269739960791,23253035475]

# futhark flops-memops.fut script
flops_OptT_double = {   
    '0_UNIFORM_1000': 9789626000,
    '0_UNIFORM_3000': 29368878000,
    '0_UNIFORM_5000':   48948130000,
    '0_UNIFORM_100000': 978962600000,
    '1_RAND_100000': 1007189668136,
    '1_RAND_NORMH_100000': 1011525907312,
    '1_RAND_NORMW_100000': 1011650780036,
    '4_SKEWED_1_100000': 52405527544,
    '4_SKEWED_INV_1_100000': 21913746500
}

flops_OptsTB_double = flops_OptT_double

# 1_RAND_100000, 4_SKEWED_1_100000
# flops_OptT_double = [264062072453,28486850361]
# flops_OptsTB_double = [393584980737,34581179352]

flops_OptT_single = {k: int(v*0.7) for k, v in flops_OptT_double.items()}
flops_OptsTB_single = {k: int(v*0.7) for k, v in flops_OptT_double.items()}

# Memory Accesses
mem_accesses = [2738946048,417930000,82836705060,12709138884,8083858200,1243743444]
# double_mem_access = [5477892096,835860000,165673410120,25418277768,16167716400,2487486888]
# 125066900000
# 250133800000
# PS C:\Work\GitHub\wojciechpawlak\diku.OptionsPricing\futhark> cat D:\data\fixed_rate\0_UNIFORM_1000.in | .\flops-memops.exe
# 1250669000
# 2501338000
# PS C:\Work\GitHub\wojciechpawlak\diku.OptionsPricing\futhark> cat D:\data\fixed_rate\1_RAND_100000.in | .\flops-memops.exe
# 128411760092
# 256823520184
# PS C:\Work\GitHub\wojciechpawlak\diku.OptionsPricing\futhark> cat D:\data\fixed_rate\2_RANDCONSTHEIGHT_100000.in | .\flops-memops.exe
# 22841348000
# 45682696000
# PS C:\Work\GitHub\wojciechpawlak\diku.OptionsPricing\futhark> cat D:\data\fixed_rate\4_SKEWED_1_100000.in | .\flops-memops.exe
# 13137486512
# 26274973024
# PS C:\Work\GitHub\wojciechpawlak\diku.OptionsPricing\futhark> cat D:\data\fixed_rate\4_SKEWED_5_100000.in | .\flops-memops.exe
# 28002818612
# 56005637224

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
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 5 + height, text, ha=ha[xpos], va='bottom', rotation=rot)

def dumpclean(obj):
    if type(obj) == dict:
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                print(k, end=' ')
                dumpclean(v)
            else:
                print('%s : %s' % (k, v), end=' ')
            print()
    elif type(obj) == list:
        for v in obj:
            if hasattr(v, '__iter__'):
                dumpclean(v)
            else:
                print(v, end=' ')
    else:
        print(
            obj, end=' ')

def gather_gpu_results(filename, input_path, datasets, limited_datasets_count, device_results_filenames):
    results_dict = dict()
    for filename in device_results_filenames:
        with open(input_path + filename) as f:
            lines = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            # lines = [x.strip().split(' ') for x in [ line for line in lines 
                # if '2018-'in line and not 'Expected' in line and not 'getLastCudaError()' in line]]
            lines = [x.strip().split(delimeter) for x in [ line for line in lines 
                if line.strip() and not 'file' in line and not 'terminate' in line and not 'what()' in line]]

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
                if not limited_datasets_count > 0: # and exists(line[10]):
                    dataset = splitext(basename(line[10]))[0]
                    datasets.add(dataset)

            kernels_list = list(kernels)
            kernels_list.sort(key=itemgetter(4,1,2,3))

            # datasets = sorted(datasets)
            datasets_count = len(datasets)
            timings = [[sys.maxsize,sys.maxsize]] * datasets_count
            datesets_iter = iter(datasets)

            # assign result per dataset to each of kernels
            results_dict.update(dict((kernel[0], timings.copy()) for kernel in kernels))
            for line in lines:
                for key in kernels_list:
                    if key[1] == int(line[3]) and key[2] == line[5] and key[3] == int(line[4]) and key[4] == line[1] and key[5] == device and key[6] == version: # kernel, sort, blocksize, precision match
                        # print(version + ' ' + device + ' ' + str(line) + ' ' + key[0])
                        timing = 0
                        memsize = 0
                        try:
                            timing = int(line[7]) # total time
                            memsize = int(line[8]) # total time
                        except ValueError:
                            timing = 0
                            memsize = 0
                        # if exists(line[10]):
                        # dataset = splitext(basename(line[10]))[0]
                        dataset = line[0]
                        index = 0
                        for index_dataset in datasets:
                            if dataset == index_dataset:
                                results_dict[key[0]][index] = (timing, memsize)
                                # if len(results_dict[key[0]][index]) == 0:
                                #     results_dict[key[0]][index] = [(timing, memsize)]
                                # else:
                                #     results_dict[key[0]][index].append((timing, memsize))
                                break
                            else:
                                index += 1
                        break
    return results_dict

def gather_cpu_results(cpu_results_path, datasets):
    # get CPU results as a reference baseline for speedups
    # double_cpu_results = [432, 80, 9828, 1668, 720, 115, 2180, 409, 1195, 205, 968, 162, 854, 151]
    # single_cpu_results = [341, 74, 16847, 2990, 1257, 214, 2513, 423, 1449, 336, 1058, 170, 1006, 168] # in ms

    single_cpu_results = [sys.maxsize] * len(datasets)
    double_cpu_results = [sys.maxsize] * len(datasets)
    with open(cpu_results_path) as f:
        lines = f.readlines()
        cpu_delimeter = ',' if ',' in lines[0] else ' ' 
        lines = [x.strip().split(cpu_delimeter) for x in [ line for line in lines 
            if not 'file' in line 
                and 'double' in line or 'single' in line or 'float' in line and not 'Expected' in line and not 'getLastCudaError()' in line]]

        for line in lines:
            dataset = line[0]
            index = 0
            for index_dataset in datasets:
                if dataset == index_dataset:
                    if line[0] == 'single': # fincalc format
                        single_cpu_results[index] = int(line[2])
                    elif line[0] == 'double': # fincalc format
                        double_cpu_results[index] = int(line[2])
                    elif line[1] == 'float':
                        single_cpu_results[index] = int(line[7])
                    elif line[1] == 'double':
                        double_cpu_results[index] = int(line[7])
                    else:
                        raise Exception("Wrong precision")
                    break
                else:
                    index += 1
    return (single_cpu_results, double_cpu_results)
class Plotter:
    args = ''
    results_dict = []
    datasets = []
    datasets_count = -1
    output_path = ''

    # offset = [-1, 0, 1]
    # width = 0.3  # the width of the bars

    # colors = ['DarkBlue', 'SlateBlue', 'Orchid', 'PaleVioletRed', 'Orange']
    # colors = ['SkyBlue', 'IndianRed', 'Green', 'Orange']
    colors = ['#fd4445', '#ffaf00', '#0091cf', '#50c878']
    offsets = [-1, 0, 1]
    # offsets = [-2, -1, 0, 1, 2]
    # offsets = [-1.5, -0.5, 0.5, 1.5]
    width = 0.2  # the width of the bars

    def __init__(self, args, results_dict, datasets, datasets_count, output_path):
        self.args = args
        self.results_dict = results_dict
        self.datasets = datasets
        self.datasets_count = datasets_count
        self.output_path = output_path

    def plot0(self):
        for precision in precision_dict.values():
            runtime_fig, runtime_ax = plt.subplots()
            runtime_rects = []

            inds = np.arange(datasets_count)  # the x locations for the groups

            runtime_ax.set_ylabel('Execution Time [ms]')
            runtime_ax.set_title('Comparison of best execution time across datasets, devices and versions (' + precision + ' precision)')
            runtime_ax.set_xticks(inds)
            runtime_ax.set_xticklabels(datasets)

            speedup_fig, speedup_ax = plt.subplots()
            speedup_rects = []

            speedup_ax.set_ylabel('Speedup')
            speedup_ax.set_title('Comparison of best speedups across datasets, devices and versions (' + precision + ' precision)')
            speedup_ax.set_xticks(inds)
            speedup_ax.set_xticklabels(datasets)
            
            count = 0
            for device in devices:
                for version in version_dict.values():
                    filtered_results_dict = {k:v for (k,v) in results_dict.items() if precision in k and version in k and device in k}
                    filtered_kernel_names = np.array(list(filtered_results_dict.keys())).tolist()
                    filtered_best_results = []
                    filtered_best_results_memsize = []
                    filtered_best_speedups = []
                    
                    for dataset_index in range(datasets_count):
                        lst = np.array(list(filtered_results_dict.values()))[:,dataset_index].tolist()
                        filtered_kernel_timings, filtered_kernel_memsize = zip(*lst)
                        filtered_kernel_timings = list(filtered_kernel_timings)
                        filtered_kernel_memsize = list(filtered_kernel_memsize)
                        best_result = int(round(min(filtered_kernel_timings)/1000))
                        filtered_best_results.append(best_result)
                        best_result_index = filtered_kernel_timings.index(min(filtered_kernel_timings))
                        cpu_result = single_cpu_results[dataset_index] if precision == 'Single' else double_cpu_results[dataset_index]
                        speedup = cpu_result/best_result
                        filtered_best_speedups.append(speedup)
                        best_result_memsize = filtered_kernel_memsize[best_result_index]/(1024*1024)
                        filtered_best_results_memsize.append(best_result_memsize)

                        print(datasets[dataset_index] + ' ' +
                            str(best_result) + ' ms ' +
                            "{0:0.3f}".format(best_result_memsize) + ' MB ' +
                            "{0:0.2f}".format(speedup) + 'x ' +
                            filtered_kernel_names[best_result_index])

                    flops = []
                    if precision == precision_dict['float'] and version == version_dict['cuda-option']: flops = flops_OptT_single
                    if precision == precision_dict['float'] and version == version_dict['cuda-multi']: flops = flops_OptsTB_single
                    if precision == precision_dict['double'] and version == version_dict['cuda-option']: flops = flops_OptT_double
                    if precision == precision_dict['double'] and version == version_dict['cuda-multi']: flops = flops_OptsTB_double

                    print(filtered_best_results)

                    gflops_results = [a*(10e-9)/(b*(10e-3)) for a,b in zip(flops,filtered_best_results)]
                    truncated_gflops = ["{0:0.2f}".format(s) for s in gflops_results]
                    gflops_str = ' & '.join(truncated_gflops)
                    print("GFLOP/s: " + gflops_str)
                    
                    # Do not calcualte memory bandwidth
                    # num_bytes = 4 if precision == 'Single' else 8
                    # memband_results = [(a*num_bytes)*(10e-9)/(b*(10e-3)) for a,b in zip(mem_accesses,filtered_best_results)]
                    # truncated_memband = ["{0:0.2f}".format(s) for s in memband_results]
                    # memband_str = ' & '.join(truncated_memband)
                    # print("Mem bandwidth GB/s: " + memband_str)

                    print([float("{0:0.2f}".format(s)) for s in filtered_best_speedups])
                    truncated_speedups = ["{0:0.2f}$\\times$".format(s) for s in filtered_best_speedups]
                    speedup_str = ' '.join(truncated_speedups)
                    print(speedup_str)

                    truncated_memsize_results = ["{0:0.3f}".format(s) for s in filtered_best_results_memsize]
                    results_str = ' '.join(truncated_memsize_results)
                    print(results_str)

                    print()

                    label = device + " " + version
                    
                    kernel_rects = runtime_ax.bar(inds + offset[count]*width, filtered_best_results, width, color=colors[count], label=label)
                    runtime_rects.append(kernel_rects)

                    kernel_rects = speedup_ax.bar(inds + offset[count]*width, filtered_best_speedups, width, color=colors[count], label=label)
                    speedup_rects.append(kernel_rects)

                    count = count + 1

            rect_heights = []
            for kernel_rects in runtime_rects:
                for rect in kernel_rects:
                    rect_heights.append(rect.get_height())
            max_rect_height = max(rect_heights)
            
            for kernel_rects in runtime_rects: 
                autolabel(runtime_ax, kernel_rects, 'center', 0.25, 0.9*max_rect_height, 45)

            runtime_ax.legend()
            # plt.show()
            runtime_fig.set_size_inches(args.height, args.width)
            output_path = output_path_dir + '\\figures\\' + 'best_runtimes_' + precision + '.' + args.figure_format
            runtime_fig.savefig(output_path, format=args.figure_format, dpi=1200, bbox_inches='tight')

            rect_heights = []
            for kernel_rects in speedup_rects:
                for rect in kernel_rects:
                    rect_heights.append(rect.get_height())
            max_rect_height = max(rect_heights)
            
            for kernel_rects in speedup_rects: 
                autolabel(speedup_ax, kernel_rects, 'center', 0.25, 0.9*max_rect_height, 90)

            speedup_ax.legend()
            # plt.show()
            speedup_fig.set_size_inches(args.height, args.width)
            output_path = output_path_dir + '\\figures\\' + 'best_speedups_' + precision + '.' + args.figure_format
            speedup_fig.savefig(output_path, format=args.figure_format, dpi=1200, bbox_inches='tight')

    def plot1(self):
        for precision in precision_dict.values():
            for device in devices:
                for version in version_dict.values():
                    runtime_fig, runtime_ax = plt.subplots()
                    runtime_rects = []

                    inds = np.arange(datasets_count)  # the x locations for the groups

                    runtime_ax.set_ylabel('Slowdown')
                    runtime_ax.set_title('Comparison of optimization impact across datasets (' + device + ', ' + version + ', ' + precision + ' precision)')
                    runtime_ax.set_xticks(inds)
                    runtime_ax.set_xticklabels(datasets)

                    count = 0
                    filtered_best_results = []
                    for optimization in optimization_dict.keys():
                        filtered_results_dict = {k:v for (k,v) in results_dict.items() if precision in k and version in k and device in k and optimization in k}
                        filtered_kernel_names = np.array(list(filtered_results_dict.keys())).tolist()
                        filtered_best_names = []
                        optimization_best_results = []

                        for dataset_index in range(datasets_count):
                            filtered_kernel_timings = np.array(list(filtered_results_dict.values()))[:,dataset_index].tolist()
                            best_result = int(round(min(filtered_kernel_timings)[0]/1000))
                            optimization_best_results.append(best_result)
                            best_result_index = filtered_kernel_timings.index(min(filtered_kernel_timings))
                            cpu_result = single_cpu_results[dataset_index] if precision == 'single' else double_cpu_results[dataset_index]
                            speedup = cpu_result/best_result
                            name = filtered_kernel_names[best_result_index]
                            filtered_best_names.append(name)
                            print(datasets[dataset_index] + ' ' +
                                str(best_result) + ' ms ' +
                                "{0:0.2f}".format(speedup) + 'x ' +
                                name)

                        print(optimization_best_results)
                        if optimization == '':
                            filtered_best_results = optimization_best_results.copy()
                        else:
                            results = [a/b for a,b in zip(optimization_best_results,filtered_best_results)]
                            
                            kernels_length = len(filtered_results_dict)

                            label = optimization_dict[optimization]
                            
                            kernel_rects = runtime_ax.bar(inds + offset[count]*width, results, width, color=colors[count], label=label)
                            runtime_rects.append(kernel_rects)

                            count = count + 1

                    rect_heights = []
                    for kernel_rects in runtime_rects:
                        for rect in kernel_rects:
                            rect_heights.append(rect.get_height())
                    max_rect_height = max(rect_heights)
                    
                    for kernel_rects in runtime_rects: 
                        autolabel(runtime_ax, kernel_rects, 'center', 0.25, 0.9*max_rect_height, 45)

                    runtime_ax.legend()
                    # plt.show()
                    runtime_fig.set_size_inches(args.height, args.width)
                    version_key = list(version_dict.keys())[list(version_dict.values()).index(version)]

                    output_path = output_path_dir + '\\figures\\' + 'opt_impact_' + precision + '_' + device + '_' + version_key + '.' + args.figure_format
                    runtime_fig.savefig(output_path, format=args.figure_format, dpi=1200, bbox_inches='tight')

        # for key, value in results_dict.items():
        #     value.sort(key=itemgetter(0))

    def plot2(self):
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
                    kernel_timings = np.array(list(filtered_results_dict.values()))[:,dataset_index].tolist()
                    kernel_timings = [item for sublist in kernel_timings for item in sublist]
                    kernel_rects = ax.bar(ind, kernel_timings, width, label=kernel_names)
                    rects.append(kernel_rects)

                min_timing = min(float(s) for s in (t for t in kernel_timings if t > 0))
                max_timing = max(float(s) for s in (t for t in kernel_timings if t > 0))

            # ylim_min = min_timing
            # ylim_max = max_timing
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

                for i, v in enumerate(kernel_timings):
                # if v == 0:
                #     ax.text(ylim_min + 5, i - 0.1, 'N/A')
                # else:
                        ax.text(v + 5, i - 0.2, str(v))

                for kernel_rects in rects:
                    autolabel(kernel_rects, 'center', 0.008)

                # plt.show()
                fig.set_size_inches(args.height, args.width)
                output_path = output_path_dir + '\\figures\\'  + args.output_filename +'_' + datasets[dataset_index] + '.' + args.figure_format
                fig.savefig(output_path, format=args.figure_format, dpi=1200, bbox_inches='tight')

    def plot3(self, devices, precisions, blocksize_outer, blocksize_flat, sort_order):
        results_items = self.results_dict.items()
        filtered_results_dict = {
            k: v for (k, v) in results_items 
                if  
                    # any(p in k for p in precisions) 
                    (blocksize_outer in k or blocksize_flat in k) 
                    and any(d in k for d in devices)
                    # and devices in k
                    and any(s in k for s in sort_order)
                    # and sort_order in k
            }
        filtered_results_ordered_dict = collections.OrderedDict(sorted(filtered_results_dict.items()))

        kernel1_gflops_table_dict = dict()
        kernel2_gflops_table_dict = dict()
        device_count =len(devices)

        kernel1_highest_gflops = [0] * device_count * self.datasets_count
        kernel2_highest_gflops = [0] * device_count * self.datasets_count

        for result in filtered_results_ordered_dict.items():
            flops = []

            current_precision = precision_dict['float'] if precision_dict['float'] in result[0] else precision_dict['double']
            current_version = version_dict['cuda-option'] if version_dict['cuda-option'] in result[0] else version_dict['cuda-multi'] 

            if precision_dict['float'] == current_precision and version_dict['cuda-option'] == current_version: flops = flops_OptT_single
            if precision_dict['float'] == current_precision and version_dict['cuda-multi'] == current_version: flops = flops_OptsTB_single
            if precision_dict['double'] == current_precision and version_dict['cuda-option'] == current_version: flops = flops_OptT_double
            if precision_dict['double'] == current_precision and version_dict['cuda-multi'] == current_version: flops = flops_OptsTB_double
            
            sort_limited_dict = sort_dict.copy()
            del sort_limited_dict['-']

            key = result[0]
            for x in device_dict.values():
                key = key.replace(' ' + x,'')
            for x in precision_dict.values():
                key = key.replace(' ' + x,'')
            for x in sort_limited_dict.values():
                key = key.replace(' ' + x,' Sort')
            print_str = result[0] + ' '

            column_count = (device_count + 1)
            value = []
            if version_dict['cuda-option'] == current_version:
                value = [0] * column_count * len(result[1]) if key not in kernel1_gflops_table_dict else kernel1_gflops_table_dict[key]
            elif version_dict['cuda-multi'] == current_version:
                value = [0] * column_count * len(result[1]) if key not in kernel2_gflops_table_dict else kernel2_gflops_table_dict[key]
            else:
                raise Exception("Wrong version")

            for dataset_index in range(self.datasets_count):
                dataset = self.datasets[dataset_index]
                gflops = float(flops[dataset])*(1e-9)
                time_s = float(result[1][dataset_index][0]*(1e-6))
                gflops_per_s = gflops/time_s
                memsize = result[1][dataset_index][1]/(1024*1024*1024)
                if device_dict['gtx780'] in result[0] and precision_dict['float'] in result[0]:
                    table_index= 1 + dataset_index*column_count
                    device_index = 1 + dataset_index*device_count

                    value[table_index] = gflops_per_s if gflops_per_s > value[table_index] else value[table_index]
                    if version_dict['cuda-option'] == current_version:
                        kernel1_highest_gflops[device_index] = gflops_per_s if gflops_per_s > kernel1_highest_gflops[device_index] else kernel1_highest_gflops[device_index]
                    elif version_dict['cuda-multi'] == current_version:
                        kernel2_highest_gflops[device_index] = gflops_per_s if gflops_per_s > kernel2_highest_gflops[device_index] else kernel2_highest_gflops[device_index]

                elif device_dict['v100'] in result[0] and precision_dict['double'] in result[0]:
                    table_index= 0 + dataset_index*column_count
                    device_index = 0 + dataset_index*device_count
                    value[table_index] = gflops_per_s if gflops_per_s > value[table_index] else value[table_index]
                    if version_dict['cuda-option'] == current_version:
                        kernel1_highest_gflops[device_index] = gflops_per_s if gflops_per_s > kernel1_highest_gflops[device_index] else kernel1_highest_gflops[device_index]
                    elif version_dict['cuda-multi'] == current_version:
                        kernel2_highest_gflops[device_index] = gflops_per_s if gflops_per_s > kernel2_highest_gflops[device_index] else kernel2_highest_gflops[device_index]
                else:
                    break
                if value[2 + dataset_index*column_count] == 0: 
                    value[2 + dataset_index*column_count] = memsize
                print_str += '\t' + "{0:d}".format(flops[dataset]) + ' ' + "{0:d}".format(result[1][dataset_index][0])

            # print(print_str)        
            if version_dict['cuda-option'] == current_version:
                kernel1_gflops_table_dict[key] = value
            elif version_dict['cuda-multi'] == current_version:
                kernel2_gflops_table_dict[key] = value
            else:
                raise Exception("Wrong version")

        print(kernel1_highest_gflops)
        print(kernel2_highest_gflops)

        self.make_gflops_table(kernel1_gflops_table_dict, kernel1_highest_gflops, device_count, column_count)
        print()
        self.make_gflops_table(kernel2_gflops_table_dict, kernel2_highest_gflops, device_count, column_count)

    def make_gflops_table(self, gflops_table_dict, highest_gflops, device_count, column_count):
        count = 0
        for result in sorted(gflops_table_dict.items()):
            kernel = ''

            for k,v in kernel_dict.items():
                if v in result[0]:
                    kernel = str(k)

            version = 'o' if version_dict['cuda-option'] in result[0] else 'f'
            sort = 'ws' if 'Sort' in result[0] else 'ns'

            # print_str = result[0] + '\t'
            print_str = '$ V_{{{0}}}^{{{1}}} $'.format(version + kernel, sort)

            value = result[1]
            for dataset_index in range(self.datasets_count):
                # print_str += ' & ' + "{0:0.0f}".format(value[0 + dataset_index*3]) + ' & ' + "{0:0.0f}".format(value[1 + dataset_index*3]) + ' & ' + "{0:0.2f}".format(value[2 + dataset_index*3])
                device1_best_value = highest_gflops[0 + dataset_index*device_count]
                device2_best_value = highest_gflops[1 + dataset_index*device_count]
                device1_value = value[0 + dataset_index*column_count]
                device2_value = value[1 + dataset_index*column_count]
                device1_str = "\color[HTML]{{FE0000}} \\textbf{{{0:0.0f}}}".format(device1_value) if device1_value == device1_best_value else "{0:0.0f}".format(device1_value)
                device2_str = "\color[HTML]{{FE0000}} \\textbf{{{0:0.0f}}}".format(device2_value) if device2_value == device2_best_value else "{0:0.0f}".format(device2_value)

                print_str += ' & ' + device1_str + ' & ' + device2_str + ' & ' + "{0:0.2f}".format(value[2 + dataset_index*column_count])
            
            midrule_str = '' if (count % 2 == 0) else ' \\midrule'
                
            print(print_str + ' \\\\' + midrule_str)
            count += 1

    def plot4(self, devices, precision, blocksize_outer, blocksize_flat, sort_order, single_cpu_results, double_cpu_results, quantlib_results):
        results_items = self.results_dict.items()
        filtered_results_dict = {
            k: v for (k, v) in results_items 
                if  precision in k 
                    and (blocksize_outer in k or blocksize_flat in k) 
                    and any(d in k for d in devices)
                    # and devices in k
                    and any(s in k for s in sort_order)
                    # and sort_order in k
            }
        filtered_results_ordered_dict = collections.OrderedDict(sorted(filtered_results_dict.items()))

        gflops_table_dict = dict()
        device_count =len(devices)

        highest_gflops_device1 = {d: [0.0,0.0,0.0,0.0] for d in self.datasets} # Add QuantLib here
        highest_gflops_device2 = {d: [0.0,0.0,0.0,0.0] for d in self.datasets} # Add QuantLib here



        for result in filtered_results_ordered_dict.items():
            flops = []
            if precision == precision_dict['float'] and version_dict['cuda-option'] in result[0]: flops = flops_OptT_single
            if precision == precision_dict['float'] and version_dict['cuda-multi'] in result[0]: flops = flops_OptsTB_single
            if precision == precision_dict['double'] and version_dict['cuda-option'] in result[0]: flops = flops_OptT_double
            if precision == precision_dict['double'] and version_dict['cuda-multi'] in result[0]: flops = flops_OptsTB_double
            
            key = result[0]
            for x in device_dict.values():
                key = key.replace(' ' + x,'')
            print_str = key + ' '

            column_count = (device_count + 1)
            value = [0] * (column_count) * len(result[1]) if key not in gflops_table_dict else gflops_table_dict[key]

            for dataset_index in range(self.datasets_count):
                dataset = self.datasets[dataset_index]
                gflops = float(flops[dataset])*(1e-9)
                time_s = float(result[1][dataset_index][0]*(1e-6))
                gflops_per_s = gflops/time_s
                memsize = result[1][dataset_index][1]/(1024*1024)
                if (device_dict['gtx780'] in result[0]):
                    value[1 + dataset_index*column_count] = gflops_per_s
                elif (device_dict['v100'] in result[0]):
                    value[0 + dataset_index*column_count] = gflops_per_s
                else:
                    raise Exception("Wrong device")
                value[2 + dataset_index*column_count] = memsize
                print_str += '\t' + "{0:0.0f}".format(float(flops[dataset])) + ' ' + "{0:0.0f}".format(float(result[1][dataset_index][0]))

                highest_gflops = highest_gflops_device1 if device_dict['v100'] in result[0] else highest_gflops_device2
                # 0 - cuda-option
                # 1 - cuda-multi
                # 2 - cpu
                # 3 - QuantLib
                # compare and add best result to highest gflops
                if version_dict['cuda-option'] in key:
                    highest_gflops[dataset][0] = gflops_per_s if gflops_per_s > highest_gflops[dataset][0] else highest_gflops[dataset][0]
                elif version_dict['cuda-multi'] in key:
                    highest_gflops[dataset][1] = gflops_per_s if gflops_per_s > highest_gflops[dataset][1] else highest_gflops[dataset][1]
                else:
                    raise Exception("Wrong version")
            # print(print_str)
            gflops_table_dict[key] = value

        cpu_results = single_cpu_results if precision == precision_dict['float'] else double_cpu_results

        for dataset_index in range(self.datasets_count):
            dataset = self.datasets[dataset_index]
            result = cpu_results[dataset_index]
            gflops = float(flops[dataset])*(1e-9)
            time_s = float(result*(1e-6))
            gflops_per_s = int(gflops/time_s)
            highest_gflops_device1[dataset][2] = gflops_per_s
            highest_gflops_device2[dataset][2] = gflops_per_s

        for dataset_index in range(self.datasets_count):
            dataset = self.datasets[dataset_index]
            result = quantlib_results[dataset_index]
            gflops = float(flops[dataset])*(1e-9)
            time_s = float(result*(1e-6))
            # gflops_per_s = math.ceil(gflops/time_s)
            gflops_per_s = (gflops/time_s)
            highest_gflops_device1[dataset][3] = gflops_per_s
            highest_gflops_device2[dataset][3] = gflops_per_s

        #print(highest_gflops_device1)

        self.make_gflops_figure(highest_gflops_device1, precision, device_dict['v100'])
        self.make_gflops_figure(highest_gflops_device2, precision, device_dict['gtx780'])

    def make_gflops_figure(self, highest_gflops, precision, device):
        runtime_fig, runtime_ax = plt.subplots()
        runtime_rects = []

        inds = np.arange(self.datasets_count)  # the x locations for the groups

        runtime_ax.set_ylabel('Performance [GFLOP/s]')
        # runtime_ax.set_title('Best performance comparison across datasets (' + precision + ' precision) on ' + device)
        print('Best performance comparison across datasets (' + precision + ' precision) on ' + device)
        runtime_ax.set_xticks(inds)
        dataset_labels = [dataset_dict[d] for d in self.datasets]
        runtime_ax.set_xticklabels(dataset_labels)

        version_count = len(version_dict.values())
        count = 0
        for version_index in range(version_count):
            version = version_dict['cuda-option'] if version_index == 0 else version_dict['cuda-multi'] if version_index == 1 else version_dict['cpu'] if version_index == 2 else version_dict['ql']
            gflops_results = []
            for dataset_index in range(self.datasets_count):
                dataset = self.datasets[dataset_index]
                gflops_results.append(round(highest_gflops[dataset][version_index]))
                
            kernel_rects = runtime_ax.bar(inds + self.offsets[count]*self.width, gflops_results, self.width, color=self.colors[count], label=version)
            runtime_rects.append(kernel_rects)
            count = count + 1

        rect_heights = []
        for kernel_rects in runtime_rects:
            for rect in kernel_rects:
                rect_heights.append(rect.get_height())
        max_rect_height = max(rect_heights)
        
        for kernel_rects in runtime_rects: 
            autolabel(runtime_ax, kernel_rects, 'center', 0.15, 0.9*max_rect_height, 45)

        runtime_ax.legend(loc='upper left')
        # plt.show()
        runtime_fig.set_size_inches(self.args.height, self.args.width)

        output_path_dir = dirname(self.output_path)
        output_path = output_path_dir + '\\figures\\' + 'gflops_' + precision + '_' + device + '.' + self.args.figure_format
        runtime_fig.savefig(output_path, format=self.args.figure_format, dpi=1200, bbox_inches='tight')

def main():
    parser = ArgumentParser(description='Plot barcharts for execution results.')
    parser.add_argument('-i', '--input_path', dest='input_path', default='')
    parser.add_argument('-o', '--output_path', dest='output_path', default='')
    parser.add_argument('--plot_title', dest='plot_title', default='Kernel comparison')
    parser.add_argument('--output_filename', dest='output_filename', default='results')
    parser.add_argument('--height', type=float, dest='height', default=15)
    parser.add_argument('--width', type=float, dest='width', default=15)
    parser.add_argument('-d', '--dataset', type=int, dest='dataset', default=0)
    parser.add_argument('--datasets', dest='datasets', nargs='+', default=' ')
    # parser.add_argument('-k', '--kernel_type', dest='kernel_type', default=' ')
    parser.add_argument('-k','--kernel_types', dest='kernel_types', nargs='+', default=' ')
    parser.add_argument('-p', '--plots', dest='plots', nargs='+', default=[3,4])
    parser.add_argument('-f', '--figure_format', dest='figure_format', default='pdf')

    args = parser.parse_args()
    args.plots = [int(i) for i in args.plots]

    config = configparser.ConfigParser()   
    config_file_path = r'C:\Work\GitHub\wojciechpawlak\diku.OptionsPricing\results_v100\research_barcharts_diku.config'
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

    cpu_results_filenames = json.loads(config['Paths']['cpu_results'])
    device_results_filenames = json.loads(config['Paths']['device_results'])

    # get data
    # print(input_path)
    if not os.path.isfile(input_path):
        exit

    limited_datasets_path = json.loads(config['Paths']['limited_datasets'])

    limited_datasets = []

    input_path_dir = dirname(input_path)
    output_path_dir = dirname(output_path)

    for path in limited_datasets_path:
        if isdir(path):
            filenames = [splitext(basename(f))[0] for f in listdir(path) if isfile(join(path, f)) and not 'yield.in' in f]
            limited_datasets += filenames

    if len(limited_datasets) == 0:
        limited_datasets = limited_datasets_path

    # datasets = set()
    datasets = limited_datasets if limited_datasets else set()

    datasets_plot3 = datasets.copy()
    # datasets_plot3.remove("0_UNIFORM_1000")
    datasets_plot3.remove("0_UNIFORM_3000")
    # datasets_plot3.remove("0_UNIFORM_5000")
    datasets_plot3.remove("0_UNIFORM_100000")

    devices = set()
    for device in device_dict.items():
        for filename in device_results_filenames:
            if device[0] in filename:
                devices.add(device[1])

    results_dict_plot3 = gather_gpu_results(filename, input_path, datasets_plot3, len(datasets_plot3), device_results_filenames)
    results_dict_plot4 = gather_gpu_results(filename, input_path, datasets, len(datasets), device_results_filenames)
    # dumpclean(results_dict)
    
    # average results from many trials
    # for kernel, timings in results_dict.items():
        # for dataset_index in range(datasets_count):
            # timings[dataset_index] = int(np.mean(timings[dataset_index]))

    single_cpu_results_32cores, double_cpu_results_32cores = gather_cpu_results(input_path + cpu_results_filenames[0], datasets)
    single_cpu_results_52cores, double_cpu_results_52cores = gather_cpu_results(input_path + cpu_results_filenames[1], datasets)
    # print(single_cpu_results)
    # print(double_cpu_results) 6355666
    # quantlib_results = [25720300, 75233467, 123722662, 25720300000, 32906880000, 27669909000, 33302055000, 5568996000, 564708000]
    quantlib_results = [75233467, 25720300000, 32906880000, 27669909000, 33302055000, 5568996000, 564708000]

    # print and plot the best results and speedups in comparison to OpenMP CPU code for each dataset
    plotter = Plotter(args, results_dict_plot3, datasets_plot3, len(datasets_plot3), output_path)
    if 0 in args.plots:
        plotter.plot0()
    if 1 in args.plots:
        plotter.plot1()
    if 2 in args.plots:
        plotter.plot2()
    if 3 in args.plots:
        # plotter.plot3(device_dict.values(), precision_dict['double'], '128', '512', sort_dict['-'])
        plotter.plot3(device_dict.values(), precision_dict, '128', '512', sort_dict)
        # plotter.plot3('', precision_dict['double'], '128', '512', sort_dict['-'])
        print()
        # plotter.plot3(device_dict['gtx780'], precision_dict['float'], '128', '512', sort_dict['-'])
    if 4 in args.plots:
        plotter_plot4 = Plotter(args, results_dict_plot4, datasets, len(datasets), output_path)
        plotter_plot4.plot4(device_dict.values(), precision_dict['double'], '128', '512', sort_dict, single_cpu_results_52cores, double_cpu_results_52cores, quantlib_results)
        plotter_plot4.plot4(device_dict.values(), precision_dict['float'], '128', '512', sort_dict, single_cpu_results_32cores, double_cpu_results_32cores, quantlib_results)

if __name__ == '__main__':
    main()

# flops_OptT_single = [41067741184, 6266440000, 1267815845536, 194515257232, 122466550192, 18841305952]
# flops_OptsTB_single = [41067741184, 6266440000, 1267815845536, 194515257232, 122466550192, 18841305952]
# flops_OptT_double = [43092279296, 6575360000, 1329778725560, 204021906608, 128478600128, 19766272568]
# flops_OptsTB_double = [43092279296, 6575360000, 1329778725560, 204021906608, 128478600128, 19766272568]

# 2738946048, 417930000, 82836705060, 12709138884, 8083858200, 1243743444
# 5477892096, 835860000, 165673410120, 25418277768, 16167716400, 2487486888

# flops_OptT_single = [25432489984,2880690000,749167430295,114974702569,75733038865,11660672980]
# flops_OptsTB_single = [16957166170,1976772280,468674867247,72027781445,48065154714,7427231551]
# flops_OptT_double = [36628917248,5837670080,1052751000000,169935513279,89578307165,15289858773]
# flops_OptsTB_double = [30206787127,4796889170,863636456723,138976446679,71660294257,12532714086]

# flops_OptT_double = [39260822215,6257125364,1128394529589,182145923933,96014795308,16388484074]
# flops_OptsTB_double = [32377241485,5141561013,925691500953,148962349287,76809315810,134332297159]