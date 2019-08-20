import csv,os,sys
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class CSV_Handler:
    def __init__(self, fname, data_obj, datadir = './Data/',ab=False):
        self.DATADIR = datadir
        self.fname = fname
        identifier = self.fname[:-4]
        if ab:
            self.opt_const = identifier.split('_')[1]
            self.temp = identifier.split('_')[3]
            if self.temp not in data_obj.keys():
                data_obj[self.temp] = {}
            if 'k_'+self.opt_const not in data_obj[self.temp].keys():
                data_obj[self.temp]['k_'+self.opt_const] = []
            if self.opt_const not in data_obj[self.temp].keys():
                data_obj[self.temp][self.opt_const] = []
            self.process_file(data_obj, ab=True)
        else:
            self.temp = identifier.split('_')[3]
            if self.temp not in data_obj.keys():
                data_obj[self.temp] = {}
            if 'k_e1c' not in data_obj[self.temp].keys():
                data_obj[self.temp]['k_e1c'] = []
            if 'k_e2c' not in data_obj[self.temp].keys():
                data_obj[self.temp]['k_e2c'] = []
            if 'e1c' not in data_obj[self.temp].keys():
                data_obj[self.temp]['e1c'] = []
            if 'e2c' not in data_obj[self.temp].keys():
                data_obj[self.temp]['e2c'] = []
            self.process_file(data_obj, ab=False)


    def process_file(self,data,ab=False):
        if ab:
            with open(self.DATADIR+self.fname) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:
                    k = float(row[0])
                    opt = float(row[1])
                    data[self.temp]['k_'+self.opt_const].append(k)
                    data[self.temp][self.opt_const].append(opt)
            assert len(data[self.temp]['k_'+self.opt_const]) == len(data[self.temp][self.opt_const]), 'CSV Error, length of k array not equal to length of optical constant array'
        else:
            with open(self.DATADIR+self.fname,encoding='utf-8-sig') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:
                    k = float(row[0])
                    e1c = float(row[1])
                    e2c = float(row[2])
                    data[self.temp]['k_e1c'].append(k)
                    data[self.temp]['k_e2c'].append(k)
                    data[self.temp]['e1c'].append(e1c)
                    data[self.temp]['e2c'].append(e2c)
            assert len(data[self.temp]['k_e1c']) == len(data[self.temp]['e1c']), 'CSV Error, length of k array not equal to length of optical constant array'
            assert len(data[self.temp]['k_e1c']) == len(data[self.temp]['e2c']), 'CSV Error, length of k array not equal to length of optical constant array'

def get_common_k(data_obj):
    common_k = []
    for temp in data_obj.keys():
        for opt in data_obj[temp].keys():
            if 'k_' in opt:
                old_common_k = common_k
                common_k = []
                curr_k = data_obj[temp][opt]
                if old_common_k == []:
                    old_common_k = curr_k
                for k in curr_k:
                    if k in old_common_k:
                        common_k.append(k)
    return common_k

def construct_common_k_data(data_obj, common_k):
    common_k_data = {}
    for temp in data_obj.keys():
        common_k_data[temp] = {'k':common_k}
        for opt in data_obj[temp].keys():
            if 'k_' not in opt:
                common_k_data[temp][opt] = []
                for k in common_k:
                    k_idx = data_obj[temp]['k_'+opt].index(k)
                    common_k_data[temp][opt].append(data_obj[temp][opt][k_idx])
    return common_k_data

def plot(x,y,log_x=False,log_y=False,axis=None):
    plt.plot(x,y)
    if log_x:
        plt.gca().set_xscale('log')
    if log_y:
        plt.gca().set_yscale('log')
    if axis != None:
        plt.axis(axis)

def calculate_s2(data_obj):
    k_arr = data_obj['65K']['k']
    for temp in data_obj.keys():
        data_obj[temp]['s2a'] = []
        data_obj[temp]['s2b'] = []
        for opt in data_obj[temp].keys():
            if 's1' in opt:
                axis = opt[-1]
                s1_arr = data_obj[temp][opt]
                inv_tau_arr = data_obj[temp]['tau-1'+axis]
                for i in range(len(k_arr)):
                    k = k_arr[i]
                    s1 = s1_arr[i]
                    inv_tau = inv_tau_arr[i]
                    s2 = k*s1/inv_tau
                    data_obj[temp]['s2'+axis].append(s2)

def calculate_eps(data_obj):
    k = data_obj['65K']['k']
    for temp in data_obj.keys():
        s1a = np.array(data_obj[temp]['s1a'])
        s2a = np.array(data_obj[temp]['s2a'])
        s1b = np.array(data_obj[temp]['s1b'])
        s2b = np.array(data_obj[temp]['s2b'])
        sa = s1a+1j*s2a
        sb = s1b+1j*s2b
        epsa = (376.7)*(1j)*sa/k
        epsb = (376.7)*(1j)*sb/k
        data_obj[temp]['e1a'] = np.real(epsa)
        data_obj[temp]['e2a'] = np.imag(epsa)
        data_obj[temp]['e1b'] = np.real(epsb)
        data_obj[temp]['e2b'] = np.imag(epsb)

def calculate_s2_interp(interp_fxn_obj):
    k = np.arange(100,10000,1)
    temp_interp_obj = {}
    for temp in interp_fxn_obj.keys():
        temp_interp_obj[temp] = {}
        for opt in interp_fxn_obj[temp].keys():
            if 's1' in opt:
                axis = opt[-1]
                s1 = interp_fxn_obj[temp]['s1'+axis](k)
                inv_tau = interp_fxn_obj[temp]['tau-1'+axis](k)
                s2 = k*s1/inv_tau
                temp_interp_obj[temp]['s2'+axis] = interp1d(k, s2,kind='cubic')
    for temp in temp_interp_obj.keys():
        for opt in temp_interp_obj[temp].keys():
            interp_fxn_obj[temp][opt] = temp_interp_obj[temp][opt]

def calculate_eps_interp(interp_fxn_obj):
    k = np.arange(100,10000,1)
    for temp in interp_fxn_obj.keys():
        s1a = interp_fxn_obj[temp]['s1a'](k)
        s2a = interp_fxn_obj[temp]['s2a'](k)
        s1b = interp_fxn_obj[temp]['s1b'](k)
        s2b = interp_fxn_obj[temp]['s2b'](k)
        sa = s1a+1j*s2a
        sb = s1b+1j*s2b
        epsa = (376.7)*(1j)*sa/k
        epsb = (376.7)*(1j)*sb/k
        e1a = np.real(epsa)
        e2a = np.imag(epsa)
        e1b = np.real(epsb)
        e2b = np.imag(epsb)
        interp_fxn_obj[temp]['e1a'] = interp1d(k,e1a,kind='cubic')
        interp_fxn_obj[temp]['e2a'] = interp1d(k,e2a,kind='cubic')
        interp_fxn_obj[temp]['e1b'] = interp1d(k,e1b,kind='cubic')
        interp_fxn_obj[temp]['e2b'] = interp1d(k,e2a,kind='cubic')

def plot_eps(interp_fxn_obj):
    k = np.arange(100,10000,1)
    fig,ax = plt.subplots(3,1,sharex=True)

    e1a_65k = interp_fxn_obj['65K']['e1a'](k)
    e2a_65k = interp_fxn_obj['65K']['e2a'](k)
    e1a_293k = interp_fxn_obj['293K']['e1a'](k)
    e2a_293k = interp_fxn_obj['293K']['e2a'](k)

    e1b_65k = interp_fxn_obj['65K']['e1b'](k)
    e2b_65k = interp_fxn_obj['65K']['e2b'](k)
    e1b_293k = interp_fxn_obj['293K']['e1b'](k)
    e2b_293k = interp_fxn_obj['293K']['e2b'](k)

    e1c_65k = interp_fxn_obj['65K']['e1c'](k)
    e2c_65k = interp_fxn_obj['65K']['e2c'](k)
    e1c_293k = interp_fxn_obj['293K']['e1c'](k)
    e2c_293k = interp_fxn_obj['293K']['e2c'](k)

    plt.suptitle('YBCO $\epsilon(\omega)$ Values')

    ax[0].plot(k,e1a_65k)
    ax[0].plot(k,e2a_65k)
    ax[0].plot(k,e1a_293k)
    ax[0].plot(k,e2a_293k)
    ax[0].legend(['$\epsilon_{1a}$ 65K','$\epsilon_{2a}$ 65K', '$\epsilon_{1a}$ 293K', '$\epsilon_{2a}$ 293K'])
    ax[0].set_xscale('log')

    ax[1].plot(k,e1b_65k)
    ax[1].plot(k,e2b_65k)
    ax[1].plot(k,e1b_293k)
    ax[1].plot(k,e2b_293k)
    ax[1].legend(['$\epsilon_{1b}$ 65K','$\epsilon_{2b}$ 65K', '$\epsilon_{1b}$ 293K', '$\epsilon_{2b}$ 293K'])


    ax[2].plot(k,e1c_65k)
    ax[2].plot(k,e2c_65k)
    ax[2].plot(k,e1c_293k)
    ax[2].plot(k,e2c_293k)
    ax[2].legend(['$\epsilon_{1c}$ 65K','$\epsilon_{2c}$ 65K', '$\epsilon_{1c}$ 293K', '$\epsilon_{2c}$ 293K'])
    plt.xlabel('k (cm$^{-1}$)')

    plt.xlim(700,1000)
    plt.ylim(-100,100)

    plt.show()

def create_datasheet(interp_fxn_obj,temp,fname):
    k = np.arange(100,10000,1)
    e1a = interp_fxn_obj[temp]['e1a'](k)
    e2a = interp_fxn_obj[temp]['e2a'](k)
    e1b = interp_fxn_obj[temp]['e1b'](k)
    e2b = interp_fxn_obj[temp]['e2b'](k)
    e1c = interp_fxn_obj[temp]['e1c'](k)
    e2c = interp_fxn_obj[temp]['e2c'](k)

    e1ab = (e1a+e1b)/2
    e2ab = (e2a+e2b)/2

    with open('./Data/'+fname, mode='w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        i=0
        for i in range(len(k)):
            row = [k[i], e1ab[i], e2ab[i], e1c[i], e2c[i]]
            writer.writerow(row)
            i+=1



def main():
    total_data = {}
    interp_fxn = {}
    csvh = CSV_Handler('YBCO_s1a_y6.75_65K.csv', total_data, ab=True)
    csvh = CSV_Handler('YBCO_s1a_y6.75_293K.csv', total_data, ab=True)
    csvh = CSV_Handler('YBCO_s1b_y6.75_65K.csv', total_data, ab=True)
    csvh = CSV_Handler('YBCO_s1b_y6.75_293K.csv', total_data, ab=True)
    csvh = CSV_Handler('YBCO_tau-1a_y6.75_65K.csv', total_data, ab=True)
    csvh = CSV_Handler('YBCO_tau-1a_y6.75_293K.csv', total_data, ab=True)
    csvh = CSV_Handler('YBCO_tau-1b_y6.75_65K.csv', total_data, ab=True)
    csvh = CSV_Handler('YBCO_tau-1b_y6.75_293K.csv', total_data, ab=True)
    csvh = CSV_Handler('YBCO_all_y6.95_65K.csv', total_data, ab=False)
    csvh = CSV_Handler('YBCO_all_y6.95_293K.csv', total_data, ab=False)

    for temp in total_data.keys():
        print(temp)
        interp_fxn[temp] = {}
        for opt in total_data[temp].keys():
            if 'k_' not in opt:
                print('\tInterpolating '+opt)
                x = total_data[temp]['k_'+opt]
                y = total_data[temp][opt]
                interp_fxn[temp][opt] = interp1d(x,y, kind='cubic')

    calculate_s2_interp(interp_fxn)
    calculate_eps_interp(interp_fxn)
    plot_eps(interp_fxn)
    create_datasheet(interp_fxn, '65K', 'YBCO_e1ab_e2ab_e1c_e2c_65K_interp.csv')
    create_datasheet(interp_fxn, '293K', 'YBCO_e1ab_e2ab_e1c_e2c_293K_interp.csv')

    """
    common_k = get_common_k(total_data)
    common_k_data = construct_common_k_data(total_data, common_k)


    for temp in common_k_data.keys():
        k = np.array(common_k_data[temp]['k'])
        e1a = np.array(common_k_data[temp]['e1a'])
        e2a = np.array(common_k_data[temp]['e2a'])
        e1b = np.array(common_k_data[temp]['e1b'])
        e2b = np.array(common_k_data[temp]['e2b'])
        e1c = np.array(common_k_data[temp]['e1c'])
        e2c = np.array(common_k_data[temp]['e2c'])

        e1c = 4
        e2c = 0.1

        e1ab = (e1a+e1b)/2
        e2ab = (e2a+e2b)/2
        with open('./Data/YBCO_e1ab_e2ab_ec_const_'+temp+'.csv', mode='w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i in range(len(k)):
                row=[k[i],e1ab[i],e2ab[i],e1c,e2c]
                writer.writerow(row)

    plot_bool=False
    if plot_bool:
        k = common_k_data['65K']['k']
        plt.figure()
        plot(k, common_k_data['65K']['e1c'], log_x=True)
        plot(k, common_k_data['65K']['e2c'], log_x=True)
        plot(k, common_k_data['293K']['e1c'], log_x=True)
        plot(k, common_k_data['293K']['e2c'], log_x=True)
        plt.show()
    """

if __name__=='__main__': main()
