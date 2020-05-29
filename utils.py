import csv,os
import numpy as np
from scipy.interpolate import interp1d

class CSV_Handler:
    def __init__(self, fname, temp, data_obj,
                    datadir = './Data/', delimiter=",",
                    opt_consts=[], include_k=True):
        self.DATADIR = datadir
        self.fname = fname
        self.temp = temp
        self.delimiter = delimiter
        self.include_k = include_k
        if self.temp not in data_obj.keys():
            data_obj[self.temp] = {}
        for opt_const in opt_consts:
            if include_k:
                if 'k_'+opt_const not in data_obj[self.temp].keys():
                    data_obj[self.temp]['k_'+opt_const] = []
            if opt_const not in data_obj[self.temp].keys():
                data_obj[self.temp][opt_const] = []
        self.process_file(data_obj, opt_consts)

    def process_file(self,data,opt_consts=[]):
        with open(os.path.join(self.DATADIR,self.fname), mode='r', encoding='utf-8-sig') as csvfile:
            print("Reading data from {}".format(os.path.join(self.DATADIR,self.fname)))
            csvreader = csv.reader(csvfile, delimiter=self.delimiter)
            for row in csvreader:
                i=0
                try:
                    if self.include_k:
                        k = float(row[0])
                        i+=1
                    for opt_const in opt_consts:
                        opt = float(row[i])
                        if self.include_k:
                            data[self.temp]['k_'+opt_const].append(k)
                        data[self.temp][opt_const].append(opt)
                        if self.include_k:
                            assert len(data[self.temp]['k_'+opt_const]) == len(data[self.temp][opt_const]), 'CSV Error, length of k array not equal to length of optical constant array'
                        i+=1
                except ValueError:
                    pass

def create_datasheet(source="",data_dir="./collated_data/",\
                        temp="300K",material="Bi2212",ks=[],\
                        opts=[],interp_fxn_obj={},delimiter=","):
    fname = "_".join([source,material,"k","_".join(opts),temp])+".dat"
    with open(os.path.join(data_dir,fname), mode='w') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        for k in ks:
            row = [k]
            for opt in opts:
                row.append(interp_fxn_obj[temp][opt](k))
            writer.writerow(row)
    print("Created datasheet for {} (T={}) at {}".format(opts,temp,os.path.join(data_dir,fname)))

def interpolate(total_data):
    interp_fxn_obj = {}
    for temp in total_data.keys():
        print(temp)
        interp_fxn_obj[temp] = {}
        for opt in total_data[temp].keys():
            if 'k_' not in opt:
                print('\tInterpolating '+opt)
                x = total_data[temp]['k_'+opt]
                y = total_data[temp][opt]
                interp_fxn_obj[temp][opt] = interp1d(x,y, kind='cubic', fill_value=(y[0],y[-1]), bounds_error=False)
    return interp_fxn_obj
