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

def plot_epsab_epsc(k,suptitle,interp_fxn_obj):
    fig,ax = plt.subplots(2,1,sharex=True)

    e1ab_10k = interp_fxn_obj['10K']['e1ab'](k)
    e2ab_10k = interp_fxn_obj['10K']['e2ab'](k)
    e1ab_100k = interp_fxn_obj['100K']['e1ab'](k)
    e2ab_100k = interp_fxn_obj['100K']['e2ab'](k)

    e1c_10k = interp_fxn_obj['10K']['e1c'](k)
    e2c_10k = interp_fxn_obj['10K']['e2c'](k)
    e1c_100k = interp_fxn_obj['100K']['e1c'](k)
    e2c_100k = interp_fxn_obj['100K']['e2c'](k)

    plt.suptitle(suptitle)

    ax[0].plot(k,e1ab_10k)
    ax[0].plot(k,e2ab_10k)
    ax[0].plot(k,e1ab_100k)
    ax[0].plot(k,e2ab_100k)
    ax[0].legend(['$\epsilon_{1ab}$ 10K','$\epsilon_{2ab}$ 10K', '$\epsilon_{1ab}$ 100K', '$\epsilon_{2ab}$ 100K'])
    ax[0].set_xscale('log')

    ax[1].plot(k,e1c_10k)
    ax[1].plot(k,e2c_10k)
    ax[1].plot(k,e1c_100k)
    ax[1].plot(k,e2c_100k)
    ax[1].legend(['$\epsilon_{1c}$ 10K','$\epsilon_{2c}$ 10K', '$\epsilon_{1c}$ 100K', '$\epsilon_{2c}$ 100K'])
    plt.xlabel('k (cm$^{-1}$)')

    plt.show()

def plot_eps(k, interp_fxn_obj):
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

def calculate_epsab_interp(interp_fxn_obj):
    k = np.arange(100,10000,1)
    for temp in interp_fxn_obj.keys():
        e1a = interp_fxn_obj[temp]['e1a']
        e2a = interp_fxn_obj[temp]['e2a']
        e1b = interp_fxn_obj[temp]['e1b']
        e2b = interp_fxn_obj[temp]['e2b']
        e1ab = (e1a+e1b)/2
        e2ab = (e2a+e2b)/2
        interp_fxn_obj[temp]['e1ab'] = interp1d(k,e1ab,kind='cubic')
        interp_fxn_obj[temp]['e2ab'] = interp1d(k,e2ab,kind='cubic')

def create_datasheet(fname,material,temp,k,opt_consts,interp_fxn_obj):
    opt_const_values = []

    for opt in opt_consts:
        opt_const_values.append(interp_fxn_obj[temp][opt](k))

    with open('./Collated_Data/'+fname, mode='w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        i=0
        for i in range(len(k)):
            row = [k[i]]
            for opt_values in opt_const_values:
                row.append(opt_values[i])
            writer.writerow(row)
            i+=1

def print_interp_fxn(interp_fxn_obj):
    for temp in interp_fxn_obj.keys():
        print(temp)
        for opt in interp_fxn_obj[temp].keys():
            print('\t'+opt)

def YBCO_analysis():
    total_data = {}
    interp_fxn = {}

    csvh = CSV_Handler('YBCO_s1a_s2a_y6.95_10K.csv', total_data, opt_consts = ['s1a','s2a'])
    csvh = CSV_Handler('YBCO_s1a_s2a_y6.95_100K.csv', total_data, opt_consts = ['s1a','s2a'])
    csvh = CSV_Handler('YBCO_s1b_s2b_y6.95_10K.csv', total_data, opt_consts = ['s1b','s2b'])
    csvh = CSV_Handler('YBCO_s1b_s2b_y6.95_100K.csv', total_data, opt_consts = ['s1b','s2b'])
    csvh = CSV_Handler('YBCO_all_y6.95_10K.csv', total_data, opt_consts = ['e1c','e2c'])
    csvh = CSV_Handler('YBCO_all_y6.95_100K.csv', total_data, opt_consts = ['e1c','e2c'])

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
    calculate_epsab_interp(interp_fxn)
    k = np.arange(101,10000,1)
    plot_eps(k, interp_fxn)
    #create_datasheet(interp_fxn, '10K', 'YBCO_e1ab_e2ab_e1c_e2c_10K_interp_ec_const.csv','YBCO')
    #create_datasheet(interp_fxn, '100K', 'YBCO_e1ab_e2ab_e1c_e2c_100K_interp_ec_const.csv','YBCO')

def BSCCO_analysis():
    total_data_BSCCO = {}
    interp_fxn_BSCCO = {}

    csvh = CSV_Handler('Basov_BSCCO2212_10K.csv', total_data_BSCCO, opt_consts = ['e1ab','e2ab'])
    csvh = CSV_Handler('Basov_BSCCO2212_100K.csv', total_data_BSCCO, opt_consts = ['e1ab','e2ab'])
    csvh = CSV_Handler('Bi2212_fit_e1_10K.csv', total_data_BSCCO, opt_consts = ['e1c'])
    csvh = CSV_Handler('Bi2212_fit_e2_10K.csv', total_data_BSCCO, opt_consts = ['e2c'])
    csvh = CSV_Handler('Bi2212_fit_e1_100K.csv', total_data_BSCCO, opt_consts = ['e1c'])
    csvh = CSV_Handler('Bi2212_fit_e2_100K.csv', total_data_BSCCO, opt_consts = ['e2c'])

    for temp in total_data_BSCCO.keys():
        print(temp)
        interp_fxn_BSCCO[temp] = {}
        for opt in total_data_BSCCO[temp].keys():
            if 'k_' not in opt:
                print('\tInterpolating '+opt)
                x = total_data_BSCCO[temp]['k_'+opt]
                y = total_data_BSCCO[temp][opt]
                interp_fxn_BSCCO[temp][opt] = interp1d(x,y, kind='cubic')


    k = np.arange(101,1000,1)
    opt_consts = ['e1ab','e2ab','e1c','e2c']
    plot_epsab_epsc(k,'BSCCO $\epsilon(\omega)$ Values',interp_fxn_BSCCO)
    #create_datasheet('BSCCO_e1ab_e2ab_e1c_e2c_10K_interp.csv','BSCCO','10K',k,opt_consts,interp_fxn_BSCCO)
    #create_datasheet('BSCCO_e1ab_e2ab_e1c_e2c_100K_interp.csv','BSCCO','100K',k,opt_consts,interp_fxn_BSCCO)

def DyBCO_analysis():
    total_data = {}
    interp_fxn = {}

    csvh = CSV_Handler('DyBCO_e1_e2_model_10K.csv', total_data, opt_consts = ['e1ab','e2ab'])
    csvh = CSV_Handler('DyBCO_e1_e2_model_100K.csv', total_data, opt_consts = ['e1ab','e2ab'])
    csvh = CSV_Handler('Bi2212_fit_e1_10K.csv', total_data, opt_consts = ['e1c'])
    csvh = CSV_Handler('Bi2212_fit_e2_10K.csv', total_data, opt_consts = ['e2c'])
    csvh = CSV_Handler('Bi2212_fit_e1_100K.csv', total_data, opt_consts = ['e1c'])
    csvh = CSV_Handler('Bi2212_fit_e2_100K.csv', total_data, opt_consts = ['e2c'])
    #csvh = CSV_Handler('YBCO_all_y6.95_10K.csv', total_data, opt_consts = ['e1c','e2c'])
    #csvh = CSV_Handler('YBCO_all_y6.95_100K.csv', total_data, opt_consts = ['e1c','e2c'])

    for temp in total_data.keys():
        print(temp)
        interp_fxn[temp] = {}
        for opt in total_data[temp].keys():
            if 'k_' not in opt:
                print('\tInterpolating '+opt)
                x = total_data[temp]['k_'+opt]
                y = total_data[temp][opt]
                interp_fxn[temp][opt] = interp1d(x,y, kind='cubic')

    k = np.arange(101,10000,1)
    opt_consts = ['e1ab','e2ab','e1c','e2c']
    plot_epsab_epsc(k,'DyBCO $\epsilon(\omega)$ Values', interp_fxn)

    create_datasheet('DyBCO_e1ab_e2ab_BSCCO_e1c_BSCCO_e2c_10K_interp.csv','DyBCO','10K',k,opt_consts,interp_fxn)
    create_datasheet('DyBCO_e1ab_e2ab_BSCCO_e1c_BSCCO_e2c_100K_interp.csv','DyBCO','100K',k,opt_consts,interp_fxn)


def main():
    material = 'DyBCO'
    if material=='DyBCO':
        DyBCO_analysis()
    if material=='YBCO':
        YBCO_analysis()
    if material=='BSCCO':
        BSCCO_analysis()


if __name__=='__main__': main()
