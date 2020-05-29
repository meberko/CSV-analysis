import csv,os,sys
from utils import CSV_Handler,interpolate,create_datasheet
import numpy as np
from scipy.interpolate import interp1d
from scipy import optimize
import matplotlib.pyplot as plt

def test_e1(k,a):
    return a/k**2

"""
    Calculations for conversion between different optical constants
"""

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

def calculate_eps_interp_from_sig(interp_fxn_obj):
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

def calculate_epsab_interp_from_eps(interp_fxn_obj):
    k = np.arange(100,10000,1)
    for temp in interp_fxn_obj.keys():
        e1a = interp_fxn_obj[temp]['e1a'](k)
        e2a = interp_fxn_obj[temp]['e2a'](k)
        e1b = interp_fxn_obj[temp]['e1b'](k)
        e2b = interp_fxn_obj[temp]['e2b'](k)
        e1ab = (e1a+e1b)/2
        e2ab = (e2a+e2b)/2
        interp_fxn_obj[temp]['e1ab'] = interp1d(k,e1ab,kind='cubic')
        interp_fxn_obj[temp]['e2ab'] = interp1d(k,e2ab,kind='cubic')

def calculate_eps_interp_from_n_k(ks,total_data_obj,interp_fxn_obj):
    for temp in interp_fxn_obj.keys():
        kn = total_data_obj[temp]['k_n']
        kk = total_data_obj[temp]['k_k']
        n = np.interp(ks,kn,interp_fxn_obj[temp]['n'](kn))
        k = np.interp(ks,kk,interp_fxn_obj[temp]['k'](kk))

        e1 = n**2-k**2
        e2 = 2*n*k

        interp_fxn_obj[temp]['e1'] = interp1d(ks,e1,kind='cubic')
        interp_fxn_obj[temp]['e2'] = interp1d(ks,e2,kind='cubic')

"""
    Function for creating a datasheet for input temperatures & optical constants
"""

def extrapolate(ext_range, ext_k, ext_fxn, temps, opts, total_data_obj,p0=None):
    for temp in temps:
        for opt in opts:
            x_data,y_data = [],[]
            for i,k in enumerate(total_data_obj[temp]['k_'+opt]):
                if k>ext_range[0] and k<ext_range[1]:
                    x_data.append(k)
                    y_data.append(total_data_obj[temp][opt][i])
            p,p_cov = optimize.curve_fit(ext_fxn,x_data,y_data,p0=p0)
            #print(p)
            y = ext_fxn(ext_k,*p)
            total_data_obj[temp]['k_'+opt] = np.concatenate((ext_k,total_data_obj[temp]['k_'+opt]))
            total_data_obj[temp][opt] = np.concatenate((y,total_data_obj[temp][opt]))

def print_interp_fxn(interp_fxn_obj):
    for temp in interp_fxn_obj.keys():
        print(temp)
        for opt in interp_fxn_obj[temp].keys():
            print('\t'+opt)

def clean_negatives(total_data_obj,temp,opt):
    cleaned = np.array(total_data_obj[temp][opt])
    cleaned[cleaned<0] = 0
    total_data_obj[temp][opt] = list(cleaned)

def plot_epsab_epsc(k,suptitle,interp_fxn_obj,params_LT=None, params_HT=None):
    fig,ax = plt.subplots(2,1,sharex=True)

    plt.suptitle(suptitle)

    leg = []
    for temp in interp_fxn_obj.keys():
        ax[0].plot(k,interp_fxn_obj[temp]['e1ab'](k))
        leg.append('$\epsilon_{1ab}$ '+temp)
        ax[0].plot(k,interp_fxn_obj[temp]['e2ab'](k))
        leg.append('$\epsilon_{2ab}$ '+temp)
    ax[0].legend(leg)
    ax[0].set_xscale('log')
    ax[0].set_yscale('symlog')
    if params_LT is not None:
        ax[0].plot(k,test_e1(k,params_LT[0]))
    if params_HT is not None:
        ax[0].plot(k,test_e1(k,params_HT[0]))

    leg = []
    for temp in interp_fxn_obj.keys():
        ax[1].plot(k,interp_fxn_obj[temp]['e1c'](k))
        leg.append('$\epsilon_{1c}$ '+temp)
        ax[1].plot(k,interp_fxn_obj[temp]['e2c'](k))
        leg.append('$\epsilon_{2c}$ '+temp)
    ax[1].legend(leg)
    ax[1].set_yscale('symlog')
    plt.xlabel('k (cm$^{-1}$)')

    plt.show()

def plot_eps1ab_eps1c(k,suptitle,interp_fxn_obj,params_LT=None, params_HT=None):
    fig,ax = plt.subplots(1,1)

    plt.suptitle(suptitle)
    eps1ab_RT = interp_fxn_obj["300K"]['e1ab'](k)
    eps1c_RT = interp_fxn_obj["300K"]['e1c'](k)
    eps1ab_LT = interp_fxn_obj["10K"]['e1ab'](k)
    eps1c_LT = interp_fxn_obj["10K"]['e1c'](k)
    leg = []
    ax.plot(k,eps1ab_RT)
    leg.append('$\epsilon_{1ab}$ '+"300K")
    ax.plot(k,eps1c_RT)
    leg.append('$\epsilon_{1c}$ '+"300K")
    ax.plot(k,eps1ab_LT)
    leg.append('$\epsilon_{1ab}$ '+"10K")
    ax.plot(k,eps1c_LT)
    leg.append('$\epsilon_{1c}$ '+"10K")
    ax.legend(leg)
    ax.fill_between(k,-1000,100, where=(eps1ab_RT*eps1c_RT<0), facecolor = "#f9b2ae")
    #ax.set_xscale('log')
    ax.set_yscale('symlog',linthreshy=10)
    plt.xlabel('k (cm$^{-1}$)')
    plt.show()

def plot_eps2ab_eps2c(k,suptitle,interp_fxn_obj,params_LT=None, params_HT=None):
    fig,ax = plt.subplots(1,1)

    plt.suptitle(suptitle)
    eps1ab = interp_fxn_obj["300K"]['e1ab'](k)
    eps2ab = interp_fxn_obj["300K"]['e2ab'](k)
    eps1c = interp_fxn_obj["300K"]['e1c'](k)
    eps2c = interp_fxn_obj["300K"]['e2c'](k)
    loss_ab = eps2ab/(eps1ab**2+eps2ab**2)
    loss_c = eps2c/(eps1c**2+eps2c**2)

    leg = []
    ax.plot(k, loss_ab)
    leg.append('$-Im[1/\epsilon_{1ab}]$ '+"300K")
    ax.plot(k,loss_c, color='g')
    leg.append('$-Im[1/\epsilon_{1c}]$ '+"300K")
    ax.legend(leg)
    #ax.fill_between(k,-1000,100, where=(eps1ab*eps1c<0), facecolor = "#f9b2ae")
    #ax.set_xscale('log')
    #ax.set_yscale('symlog',linthreshy=10)
    plt.xlabel('k (cm$^{-1}$)')
    plt.show()

"""
    Analysis for specific HTS and substrates
"""

def YBCO_analysis():
    total_data, interp_fxn = {},{}

    csvh = CSV_Handler('YBCO_s1a_s2a_y6.95_10K.csv','10K', total_data, opt_consts = ['s1a','s2a'])
    csvh = CSV_Handler('YBCO_s1a_s2a_y6.95_100K.csv','100K', total_data, opt_consts = ['s1a','s2a'])
    csvh = CSV_Handler('YBCO_s1b_s2b_y6.95_10K.csv','10K', total_data, opt_consts = ['s1b','s2b'])
    csvh = CSV_Handler('YBCO_s1b_s2b_y6.95_100K.csv','100K', total_data, opt_consts = ['s1b','s2b'])
    csvh = CSV_Handler('YBCO_all_y6.95_10K.csv','10K', total_data, opt_consts = ['e1c','e2c'])
    csvh = CSV_Handler('YBCO_all_y6.95_100K.csv','100K', total_data, opt_consts = ['e1c','e2c'])

    interpolate(total_data,interp_fxn)

    calculate_eps_interp_from_sig(interp_fxn)
    calculate_epsab_interp_from_eps(interp_fxn)
    k = np.arange(101,10000,1)
    opt_consts = ['e1ab','e2ab','e1c','e2c']
    plot_epsab_epsc(k,'YBCO $\epsilon(\omega)$ Values', interp_fxn)
    create = False
    if create:
        create_datasheet('YBCO_e1ab_e2ab_e1c_e2c_10K_interp_ec_const.csv','10K',k,opt_consts,interp_fxn, ec_const=True)
        create_datasheet('YBCO_e1ab_e2ab_e1c_e2c_100K_interp_ec_const.csv','100K',k,opt_consts,interp_fxn, ec_const=True)
        create_datasheet('YBCO_e1ab_e2ab_e1c_e2c_10K_interp.csv','10K',k,opt_consts,interp_fxn, ec_const=False)
        create_datasheet('YBCO_e1ab_e2ab_e1c_e2c_100K_interp.csv','100K',k,opt_consts,interp_fxn, ec_const=False)

def BSCCO_analysis():
    total_data_BSCCO, interp_fxn = {},{}

    # Import ab-plane optical constants
    csvh = CSV_Handler('Tu_2002_Bi2212_k_e1ab_6K_Drude_gap_fitted.dat', '10K', total_data_BSCCO,
        datadir="./data/BSCCO", opt_consts = ['e1ab'], delimiter=' ')
    csvh = CSV_Handler('Tu_2002_Bi2212_k_e2ab_6K_Drude_gap_fitted.dat', '10K', total_data_BSCCO,
        datadir="./data/BSCCO", opt_consts = ['e2ab'], delimiter=' ')
    csvh = CSV_Handler('Tu_2002_Bi2212_k_e1ab_100K_Drude_fitted.dat', '100K', total_data_BSCCO,
        datadir="./data/BSCCO", opt_consts = ['e1ab'], delimiter=' ')
    csvh = CSV_Handler('Tu_2002_Bi2212_k_e2ab_100K_Drude_fitted.dat', '100K', total_data_BSCCO,
        datadir="./data/BSCCO", opt_consts = ['e2ab'], delimiter=' ')
    """
        csvh = CSV_Handler('Tu_2002_Bi2212_k_e1ab_e2ab_10K.csv', '10K', total_data_BSCCO,
            datadir="./data/BSCCO", opt_consts = ['e1ab','e2ab'])
        csvh = CSV_Handler('Tu_2002_Bi2212_k_e1ab_e2ab_100K.csv', '150K', total_data_BSCCO,
            datadir="./data/BSCCO", opt_consts = ['e1ab','e2ab'])
        csvh = CSV_Handler('Tu_2002_Bi2212_k_e1ab_e2ab_295K.csv', '300K', total_data_BSCCO,
            datadir="./data/BSCCO", opt_consts = ['e1ab','e2ab'])
        csvh = CSV_Handler('Tu_2002_Bi2212_k_e1ab_10K_fitted.dat', '10K', total_data_BSCCO,
            datadir="./data/BSCCO", opt_consts = ['e1ab'],delimiter=' ')
        csvh = CSV_Handler('Tu_2002_Bi2212_k_e2ab_10K_fitted.dat', '10K', total_data_BSCCO,
            datadir="./data/BSCCO", opt_consts = ['e2ab'],delimiter=' ')
        csvh = CSV_Handler('Tu_2002_Bi2212_k_e1ab_100K_fitted.dat', '150K', total_data_BSCCO,
            datadir="./data/BSCCO", opt_consts = ['e1ab'],delimiter=' ')
        csvh = CSV_Handler('Tu_2002_Bi2212_k_e2ab_100K_fitted.dat', '150K', total_data_BSCCO,
            datadir="./data/BSCCO", opt_consts = ['e2ab'],delimiter=' ')
    """

    # Import c-axis optical constants
    csvh = CSV_Handler('Kovaleva_2004_Bi2212_k_e1c_10K_Drude_gap_fitted.dat', '10K', total_data_BSCCO,
        datadir="./data/BSCCO", opt_consts = ['e1c'], delimiter=' ')
    csvh = CSV_Handler('Kovaleva_2004_Bi2212_k_e2c_10K_Drude_gap_fitted.dat', '10K', total_data_BSCCO,
        datadir="./data/BSCCO", opt_consts = ['e2c'], delimiter=' ')
    csvh = CSV_Handler('Kovaleva_2004_Bi2212_k_e1c_100K_dielectric_fitted.dat', '100K', total_data_BSCCO,
        datadir="./data/BSCCO", opt_consts = ['e1c'], delimiter=' ')
    csvh = CSV_Handler('Kovaleva_2004_Bi2212_k_e2c_100K_dielectric_fitted.dat', '100K', total_data_BSCCO,
        datadir="./data/BSCCO", opt_consts = ['e2c'], delimiter=' ')
    """
        csvh = CSV_Handler('Tajima_1993_Bi2212_k_e1c_6K+JPR_fitted_fixed.dat', '10K', total_data_BSCCO,
            datadir="./data/BSCCO", delimiter=' ', opt_consts = ['e1c'])
        csvh = CSV_Handler('Tajima_1993_Bi2212_k_e2c_6K+JPR_fitted_fixed.dat', '10K', total_data_BSCCO,
            datadir="./data/BSCCO", delimiter=' ', opt_consts = ['e2c'])
        csvh = CSV_Handler('Kovaleva_2004_Bi2212_k_e1c_150K_fitted.dat', '150K', total_data_BSCCO,\
            datadir="./data/BSCCO", opt_consts = ['e1c'], delimiter=' ')
        csvh = CSV_Handler('Kovaleva_2004_Bi2212_k_e2c_150K_fitted.dat', '150K', total_data_BSCCO,\
            datadir="./data/BSCCO", opt_consts = ['e2c'], delimiter=' ')
        csvh = CSV_Handler('Tajima_1993_Bi2212_k_e1c_300K_fitted.dat', '300K', total_data_BSCCO,
            datadir="./data/BSCCO", delimiter=' ', opt_consts = ['e1c'])
        csvh = CSV_Handler('Tajima_1993_Bi2212_k_e2c_300K_fitted.dat', '300K', total_data_BSCCO,
            datadir="./data/BSCCO", delimiter=' ', opt_consts = ['e2c'])
        csvh = CSV_Handler('Tajima_1993_Bi2212_k_e1c_6K_JPR_delta.dat', '10K', total_data_BSCCO,
            datadir="./data/BSCCO", delimiter=' ', opt_consts = ['e1c'])
        csvh = CSV_Handler('Tajima_1993_Bi2212_k_e2c_6K_JPR_delta.dat', '10K', total_data_BSCCO,
            datadir="./data/BSCCO", delimiter=' ', opt_consts = ['e2c'])
        csvh = CSV_Handler('Kovaleva_2004_Bi2212_k_e1c_150K_fitted.dat', '150K', total_data_BSCCO,
            datadir="./data/BSCCO", delimiter=' ', opt_consts = ['e1c'])
        csvh = CSV_Handler('Kovaleva_2004_Bi2212_k_e2c_150K_fitted.dat', '150K', total_data_BSCCO,
            datadir="./data/BSCCO", delimiter=' ', opt_consts = ['e2c'])
    """

    #ext_range = [100,300]
    #ext_k = np.arange(1,99,0.1)
    #opts = ['e1ab','e2ab']
    temps = ['10K','10K']

    #extrapolate(ext_range,ext_k,test_e1,temps,opts,total_data_BSCCO,p0=[10000])
    interp_fxn = interpolate(total_data_BSCCO)

    opt_consts = ['e1ab','e2ab','e1c','e2c']
    k = np.arange(0.1,10000,0.1)
    plot_epsab_epsc(k,'BSCCO $\epsilon(\omega)$ Values',interp_fxn)
    create = False
    if create:
        create_datasheet(source="Bi2212_Tu_Drude_gap_fitted_Kovaleva_Drude_gap_fitted",\
                                data_dir="./collated_data/",\
                                temp="10K",material="Bi2212",ks=k,\
                                opts=opt_consts,interp_fxn_obj=interp_fxn,delimiter=",")
        create_datasheet(source="Bi2212_Tu_Drude_fitted_Kovaleva_dielectric_fitted",\
                                data_dir="./collated_data/",\
                                temp="100K",material="Bi2212",ks=k,\
                                opts=opt_consts,interp_fxn_obj=interp_fxn,delimiter=",")
        #create_datasheet('Bi2212_k_Tu_e1ab_e2ab_Drude_gap_fitted_Kovaleva_e1c_e2c_Drude_gap_fitted_10K.csv','10K',k,opt_consts,interp_fxn)
        #create_datasheet('Bi2212_k_Tu_e1ab_e2ab_Drude_fitted_Kovaleva_e1c_e2c_dielectric_fitted_100K.csv','100K',k,opt_consts,interp_fxn)
        #create_datasheet('BSCCO_e1ab_e2ab_e1c_e2c_10K_interp_ec_const_extrapolated_THz.csv','10K',k,opt_consts,interp_fxn, ec_const=True)
        #create_datasheet('BSCCO_e1ab_e2ab_e1c_e2c_300K_interp_ec_const_extrapolated_THz.csv','300K',k,opt_consts,interp_fxn, ec_const=True)
        #create_datasheet('Bi2212_k_Tu_e1ab_e2ab_Tajima_e1c_e2c_300K.csv','300K',k,opt_consts,interp_fxn)

def DyBCO_analysis():
    total_data,interp_fxn = {},{}

        #csvh = CSV_Handler('Bi2212_fit_e1_6K.csv', '10K', total_data, datadir="./data/DyBCO", opt_consts = ['e1c'])
    #csvh = CSV_Handler('Bi2212_fit_e2c_6K_Tajima.csv', '10K', total_data, datadir="./data/DyBCO", opt_consts = ['e2c'])
    #csvh = CSV_Handler('Bi2212_fit_e1_300K.csv', '100K', total_data, datadir="./data/DyBCO", opt_consts = ['e1c'])
    #csvh = CSV_Handler('Bi2212_fit_e2_300K.csv', '100K', total_data, datadir="./data/DyBCO", opt_consts = ['e2c'])
    #csvh = CSV_Handler('YBCO_all_y6.95_10K.csv', total_data, opt_consts = ['e1c','e2c'])
    #csvh = CSV_Handler('YBCO_all_y6.95_100K.csv', total_data, opt_consts = ['e1c','e2c'])

    interpolate(total_data,interp_fxn)

    k = np.arange(101,10000,1)
    opt_consts = ['e1ab','e2ab']
    plot_epsab_epsc(k,'DyBCO $\epsilon(\omega)$ Values', interp_fxn)
    create = False
    if create:
        create_datasheet('DyBCO_e1ab_e2ab_BSCCO_e1c_BSCCO_e2c_10K_interp.csv','10K',k,opt_consts,interp_fxn)
        create_datasheet('DyBCO_e1ab_e2ab_BSCCO_e1c_BSCCO_e2c_100K_interp.csv','100K',k,opt_consts,interp_fxn)

def LAO_analysis():
    hc = 1.24*10**(-4) # value of hc in eV*cm
    total_data = {}
    interp_fxn = {}

    csvh = CSV_Handler('LAO-n-300K.csv', '300K', total_data, opt_consts = ['n'])
    csvh = CSV_Handler('LAO-k-300K.csv', '300K', total_data, opt_consts = ['k'])
    total_data['300K']['k_n'] = list(np.array(total_data['300K']['k_n'])/(hc))
    total_data['300K']['k_k'] = list(np.array(total_data['300K']['k_k'])/(hc))

    interpolate(total_data,interp_fxn)

    ks = np.arange(1,1e4,1)
    kn = total_data['300K']['k_n']
    kk = total_data['300K']['k_k']
    n = np.interp(ks,kn,interp_fxn['300K']['n'](kn))
    k = np.interp(ks,kk,interp_fxn['300K']['k'](kk))
    e1 = n**2-k**2
    e2 = 2*n*k
    interp_fxn['300K']['e1'] = interp1d(ks,e1,kind='cubic')
    interp_fxn['300K']['e2'] = interp1d(ks,e2,kind='cubic')

    opt_consts = ['e1','e2']
    create_datasheet('LAO_e1_e2_300K.csv','300K',ks,opt_consts,interp_fxn)

def STO_analysis():
    hc = 1.24*10**(-4) # value of hc in eV*cm
    total_data = {}
    interp_fxn = {}

    """
        csvh = CSV_Handler('STO-n-300K.csv', '300K', total_data, opt_consts = ['n'])
        csvh = CSV_Handler('STO-k-300K.csv', '300K', total_data, opt_consts = ['k'])
        total_data['300K']['k_n'] = list(np.array(total_data['300K']['k_n'])/(hc))
        total_data['300K']['k_k'] = list(np.array(total_data['300K']['k_k'])/(hc))
    """
    CSV_Handler('STO_k_n_20K_45-1000_cm-1.csv', '20K', total_data, opt_consts = ['n'])
    CSV_Handler('STO_k_k_20K_45-1000_cm-1.csv', '20K', total_data, opt_consts = ['k'])
    CSV_Handler('STO_k_n_100K_45-1000_cm-1.csv', '100K', total_data, opt_consts = ['n'])
    CSV_Handler('STO_k_k_100K_45-1000_cm-1.csv', '100K', total_data, opt_consts = ['k'])
    CSV_Handler('STO_k_n_200K_45-1000_cm-1.csv', '200K', total_data, opt_consts = ['n'])
    CSV_Handler('STO_k_k_200K_45-1000_cm-1.csv', '200K', total_data, opt_consts = ['k'])
    CSV_Handler('STO_k_n_300K_45-1000_cm-1.csv', '300K', total_data, opt_consts = ['n'])
    CSV_Handler('STO_k_k_300K_45-1000_cm-1.csv', '300K', total_data, opt_consts = ['k'])

    x = total_data['20K']['k_k']
    y = total_data['20K']['k']
    plt.figure()
    plt.plot(x,y)
    plt.show()
    clean_negatives(total_data,'20K','k')
    clean_negatives(total_data,'100K','k')
    clean_negatives(total_data,'200K','k')
    clean_negatives(total_data,'300K','k')

    interpolate(total_data,interp_fxn)

    ks = np.arange(1,1e4,1)

    calculate_eps_interp_from_n_k(ks,total_data,interp_fxn)

    opt_consts = ['e1','e2']
    create_datasheet('STO_e1_e2_300K_1-10000_cm-1.csv','300K',ks,opt_consts,interp_fxn)
    create_datasheet('STO_e1_e2_200K_1-10000_cm-1.csv','200K',ks,opt_consts,interp_fxn)
    create_datasheet('STO_e1_e2_100K_1-10000_cm-1.csv','100K',ks,opt_consts,interp_fxn)
    create_datasheet('STO_e1_e2_20K_1-10000_cm-1.csv','20K',ks,opt_consts,interp_fxn)

def main():
    if len(sys.argv)<2:
        material = 'BSCCO'
    else:
        material = sys.argv[1]
    func_dict = {
        'BSCCO': BSCCO_analysis,
        'DyBCO': DyBCO_analysis,
        'YBCO': YBCO_analysis,
        'LAO': LAO_analysis,
        'STO': STO_analysis
    }
    func_dict[material]()

if __name__=='__main__': main()
