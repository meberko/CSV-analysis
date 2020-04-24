import csv,os

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
            csvreader = csv.reader(csvfile, delimiter=self.delimiter)
            for row in csvreader:
                i=0
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
