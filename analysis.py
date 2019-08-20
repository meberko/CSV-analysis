import csv,sys,os
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x,m,b):
    return x**(-m)*np.exp(b)

class CSV_Data:
    def __init__(self,fname,datadir):
        self.data = {}
        self.fname = fname
        self.datadir = datadir
        delim = input('Please input the delimiter for this data: ')
        self.ProcessCSV(delim)

    def ProcessCSV(self,delim='\t'):
        with open(self.datadir+self.fname) as csvf:
            reader = csv.reader(csvf, delimiter = delim)
            for label in reader.__next__():
                self.data[label] = []
            n_label = len(self.data.keys())
            for row in reader:
                assert len(row)==n_label,'\n\tCSV Error: Number of data != Number of labels'
                i=0
                for label in self.data.keys():
                    if row[i]!='':
                        self.data[label].append(float(row[i]))
                    else:
                        self.data[label].append(0)
                    i+=1

    def GetInputs(self):
        print('Labels: '+str(list(self.data.keys())))
        self.x_label = input('Enter name of x variable to plot: ')
        n_y = int(input('How many y variables would you like to plot?: '))
        self.y_labels = []
        for i in range(n_y):
            self.y_labels.append(input('Enter name of y variable to plot: '))
        self.xlog = input('Log scale x axis? (y/n): ')
        assert self.xlog=='y' or self.xlog=='n'
        self.ylog = input('Log scale y axis? (y/n): ')
        assert self.ylog=='y' or self.ylog=='n'

    def GetFit(self):
        x = self.data[self.x_label]
        y = self.data[self.y_labels[0]]
        #z = np.polyfit(x,y,30)
        #self.fit = np.poly1d(z)
        self.popt,self.pcov = curve_fit(func,x,y)

    def Plot(self):
        plt.figure(figsize=(6,6))
        x = self.data[self.x_label]
        for y_label in self.y_labels:
            if y_label.lower()=='e1':
                plt.plot(x,(-1)*np.array(self.data[y_label]),'r-',markersize=1)
            else:
                plt.plot(x,self.data[y_label],'ro',markersize=1)
        plt.xlabel(self.x_label)
        plt.ylabel(', '.join(self.y_labels))
        #plt.axis([10,1000,10,100000])
        if self.xlog=='y': plt.xscale('log')
        if self.ylog=='y': plt.yscale('log')

    def PlotShow(self):
        plt.show()

def main():
    argc = len(sys.argv)
    assert argc>1 and argc<4, '\n\tUsage: python3 analysis.py <filename> [dirname]'
    fname = sys.argv[1]
    if argc==3:
        dirname = sys.argv[2]
    else:
        dirname = './Data/'
    csvd = CSV_Data(fname, dirname)
    csvd.GetInputs()
    csvd.Plot()
    csvd.PlotShow()


if __name__=='__main__': main()
