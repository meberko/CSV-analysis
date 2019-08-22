import csv,os,sys
from CSV_Handler import CSV_Handler
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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
        epsa = (59.9585)*(1j)*sa/k
        epsb = (59.9585)*(1j)*sb/k
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
        epsa = (59.9585)*(1j)*sa/k
        epsb = (59.9585)*(1j)*sb/k
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

    e1a_65k = interp_fxn_obj['10K']['e1a'](k)
    e2a_65k = interp_fxn_obj['10K']['e2a'](k)
    e1a_293k = interp_fxn_obj['100K']['e1a'](k)
    e2a_293k = interp_fxn_obj['100K']['e2a'](k)

    e1b_65k = interp_fxn_obj['10K']['e1b'](k)
    e2b_65k = interp_fxn_obj['10K']['e2b'](k)
    e1b_293k = interp_fxn_obj['100K']['e1b'](k)
    e2b_293k = interp_fxn_obj['100K']['e2b'](k)

    e1c_65k = interp_fxn_obj['10K']['e1c'](k)
    e2c_65k = interp_fxn_obj['10K']['e2c'](k)
    e1c_293k = interp_fxn_obj['100K']['e1c'](k)
    e2c_293k = interp_fxn_obj['100K']['e2c'](k)

    plt.suptitle('YBCO $\epsilon(\omega)$ Values')

    ax[0].plot(k,e1a_65k)
    ax[0].plot(k,e2a_65k)
    ax[0].plot(k,e1a_293k)
    ax[0].plot(k,e2a_293k)
    ax[0].legend(['$\epsilon_{1a}$ 10K','$\epsilon_{2a}$ 10K', '$\epsilon_{1a}$ 100K', '$\epsilon_{2a}$ 100K'])
    ax[0].set_xscale('log')

    ax[1].plot(k,e1b_65k)
    ax[1].plot(k,e2b_65k)
    ax[1].plot(k,e1b_293k)
    ax[1].plot(k,e2b_293k)
    ax[1].legend(['$\epsilon_{1b}$ 10K','$\epsilon_{2b}$ 10K', '$\epsilon_{1b}$ 100K', '$\epsilon_{2b}$ 100K'])


    ax[2].plot(k,e1c_65k)
    ax[2].plot(k,e2c_65k)
    ax[2].plot(k,e1c_293k)
    ax[2].plot(k,e2c_293k)
    ax[2].legend(['$\epsilon_{1c}$ 10K','$\epsilon_{2c}$ 10K', '$\epsilon_{1c}$ 100K', '$\epsilon_{2c}$ 100K'])
    plt.xlabel('k (cm$^{-1}$)')

    plt.show()

def create_datasheet(interp_fxn_obj,temp,fname,type):
    if type=='YBCO':
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
                row = [k[i], e1ab[i], e2ab[i], 4,0.1]
                writer.writerow(row)
                i+=1
    if type=='BSCCO':
        k = np.arange(101,10000,1)
        e1ab = interp_fxn_obj[temp]['e1ab'](k)
        e2ab = interp_fxn_obj[temp]['e2ab'](k)
        e1c = interp_fxn_obj[temp]['e1c'](k)
        e2c = interp_fxn_obj[temp]['e2c'](k)
        with open('./Data/'+fname, mode='w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            i=0
            for i in range(len(k)):
                row = [k[i], e1ab[i], e2ab[i],e1c[i],e2c[i]]
                writer.writerow(row)
                i+=1

def print_interp_fxn(interp_fxn_obj):
    for temp in interp_fxn_obj.keys():
        print(temp)
        for opt in interp_fxn_obj[temp].keys():
            print('\t'+opt)

def main():
    total_data = {}
    total_data_BSCCO = {}
    interp_fxn = {}
    interp_fxn_BSCCO = {}

    csvh = CSV_Handler('YBCO_s1a_s2a_y6.95_10K.csv', total_data, opt_consts = ['s1a','s2a'])
    csvh = CSV_Handler('YBCO_s1a_s2a_y6.95_100K.csv', total_data, opt_consts = ['s1a','s2a'])
    csvh = CSV_Handler('YBCO_s1b_s2b_y6.95_10K.csv', total_data, opt_consts = ['s1b','s2b'])
    csvh = CSV_Handler('YBCO_s1b_s2b_y6.95_100K.csv', total_data, opt_consts = ['s1b','s2b'])
    csvh = CSV_Handler('YBCO_all_y6.95_10K.csv', total_data, opt_consts = ['e1c','e2c'])
    csvh = CSV_Handler('YBCO_all_y6.95_100K.csv', total_data, opt_consts = ['e1c','e2c'])

    csvh = CSV_Handler('Basov_BSCCO2212_10K.csv', total_data_BSCCO, opt_consts = ['e1ab','e2ab'])
    csvh = CSV_Handler('Basov_BSCCO2212_100K.csv', total_data_BSCCO, opt_consts = ['e1ab','e2ab'])
    csvh = CSV_Handler('Bi2212_fit_e1_10K.csv', total_data_BSCCO, opt_consts = ['e1c'])
    csvh = CSV_Handler('Bi2212_fit_e2_10K.csv', total_data_BSCCO, opt_consts = ['e2c'])
    csvh = CSV_Handler('Bi2212_fit_e1_100K.csv', total_data_BSCCO, opt_consts = ['e1c'])
    csvh = CSV_Handler('Bi2212_fit_e2_100K.csv', total_data_BSCCO, opt_consts = ['e2c'])

    """
    for temp in total_data.keys():
        print(temp)
        interp_fxn[temp] = {}
        for opt in total_data[temp].keys():
            if 'k_' not in opt:
                print('\tInterpolating '+opt)
                x = total_data[temp]['k_'+opt]
                y = total_data[temp][opt]
                interp_fxn[temp][opt] = interp1d(x,y, kind='cubic')


    calculate_eps_interp(interp_fxn)
    #plot_eps(interp_fxn)
    #create_datasheet(interp_fxn, '10K', 'YBCO_e1ab_e2ab_e1c_e2c_10K_interp_ec_const.csv','YBCO')
    #create_datasheet(interp_fxn, '100K', 'YBCO_e1ab_e2ab_e1c_e2c_100K_interp_ec_const.csv','YBCO')
    """
    for temp in total_data_BSCCO.keys():
        print(temp)
        interp_fxn_BSCCO[temp] = {}
        for opt in total_data_BSCCO[temp].keys():
            if 'k_' not in opt:
                print('\tInterpolating '+opt)
                x = total_data_BSCCO[temp]['k_'+opt]
                y = total_data_BSCCO[temp][opt]
                interp_fxn_BSCCO[temp][opt] = interp1d(x,y, kind='cubic')
    create_datasheet(interp_fxn_BSCCO, '10K', 'BSCCO_e1ab_e2ab_e1c_e2c_10K_interp.csv','BSCCO')
    create_datasheet(interp_fxn_BSCCO, '100K', 'BSCCO_e1ab_e2ab_e1c_e2c_100K_interp.csv','BSCCO')

if __name__=='__main__': main()
