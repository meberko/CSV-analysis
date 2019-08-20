import os
os.environ['KIVY_GL_BACKEND'] = 'sdl2'
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
import matplotlib.pyplot as plt
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import os,csv

class SelectionArea(GridLayout):
    def __init__(self,**kwargs):
        super(SelectionArea,self).__init__(**kwargs)
        self.datadir = './Data/'
        self.plotdir = './Plots/'
        self.delim=','
        self.data = {}

        # Layout configuration
        self.cols = 1
        self.rows = 7
        self.spacing=20

        self.CreateWidgets()
        self.CreateBindings()
        self.AddWidgets()

    # Helpers
    def CreateWidgets(self):
        # Layouts
        self.data_file_info_holder = GridLayout(cols=2,rows=1)
        self.x_value_holder = GridLayout(cols=2,rows=1)
        self.y_value_holder = GridLayout(cols=2,rows=1)
        self.title_holder = GridLayout(cols=2,rows=1)
        self.limit_holder = GridLayout(cols=4,rows=3)
        self.log_holder = GridLayout(cols=2,rows=1)
        self.plot_button_holder = GridLayout(cols=3,rows=1)

        # Labels
        self.data_fname_label = Label(text='Data file: ',halign='right',size_hint=(1,0.5))
        self.x_val_label = Label(text='X Value: ',halign='right',size_hint=(1,0.5))
        self.y_val_label = Label(text='Y Value: ',halign='right',size_hint=(1,0.5))
        self.x_min_label = Label(text='Min x value: ',halign='right',size_hint=(1,1))
        self.x_max_label = Label(text='Max x value: ',halign='right',size_hint=(1,1))
        self.y_min_label = Label(text='Min y value: ',halign='right',size_hint=(1,1))
        self.y_max_label = Label(text='Max y value: ',halign='right',size_hint=(1,1))
        self.x_label_label = Label(text='X-axis label: ',halign='right',size_hint=(1,1))
        self.y_label_label = Label(text='Y-axis label: ',halign='right',size_hint=(1,1))
        self.title_label = Label(text='Graph title: ',halign='right',size_hint=(1,0.5))

        # Text input
        self.x_min = TextInput(multiline=False, write_tab=False)
        self.x_max = TextInput(multiline=False,write_tab=False)
        self.y_min = TextInput(multiline=False,write_tab=False)
        self.y_max = TextInput(multiline=False,write_tab=False)
        self.x_label = TextInput(multiline=False,write_tab=False)
        self.y_label = TextInput(multiline=False,write_tab=False)
        self.title = TextInput(multiline=False,write_tab=False,size_hint=(1,0.5))

        # Dropdowns
        self.x_val_dropdown = DropDown()
        self.y_val_dropdown = DropDown()
        self.data_fname_dropdown = self.CreateDataFilenameDropdown(self.datadir)

        # Buttons
        self.data_fname_btn = Button(text='Please select data file.',size_hint_y=0.5,height=40)
        self.x_val_btn = Button(text='Please select x value.',size_hint_y=0.5,height=40)
        self.y_val_btn = Button(text='Please select y value.',size_hint_y=0.5,height=40)
        self.plot_btn = Button(text='Plot values!',size_hint_y=1,height=40,background_color=(0,0,255,0.8))
        self.clear_btn = Button(text='Clear plot',size_hint_y=1,height=40, background_color=(255,0,0,0.8))
        self.save_btn = Button(text='Save plot',size_hint_y=1,height=40, background_color=(0,255,0,0.8))

        # Toggle buttons
        self.x_log_btn = ToggleButton(text='Logrithmic x-axis?')
        self.y_log_btn = ToggleButton(text='Logrithmic y-axis?')

    def AddWidgets(self):
        self.data_file_info_holder.add_widget(self.data_fname_label)
        self.data_file_info_holder.add_widget(self.data_fname_btn)

        self.x_value_holder.add_widget(self.x_val_label)
        self.x_value_holder.add_widget(self.x_val_btn)

        self.y_value_holder.add_widget(self.y_val_label)
        self.y_value_holder.add_widget(self.y_val_btn)

        self.title_holder.add_widget(self.title_label)
        self.title_holder.add_widget(self.title)

        self.limit_holder.add_widget(self.x_min_label)
        self.limit_holder.add_widget(self.x_min)

        self.limit_holder.add_widget(self.y_min_label)
        self.limit_holder.add_widget(self.y_min)

        self.limit_holder.add_widget(self.x_max_label)
        self.limit_holder.add_widget(self.x_max)

        self.limit_holder.add_widget(self.y_max_label)
        self.limit_holder.add_widget(self.y_max)

        self.limit_holder.add_widget(self.x_label_label)
        self.limit_holder.add_widget(self.x_label)

        self.limit_holder.add_widget(self.y_label_label)
        self.limit_holder.add_widget(self.y_label)

        self.log_holder.add_widget(self.x_log_btn)
        self.log_holder.add_widget(self.y_log_btn)

        self.plot_button_holder.add_widget(self.clear_btn)
        self.plot_button_holder.add_widget(self.save_btn)
        self.plot_button_holder.add_widget(self.plot_btn)

        self.add_widget(self.data_file_info_holder)
        self.add_widget(self.x_value_holder)
        self.add_widget(self.y_value_holder)
        self.add_widget(self.title_holder)
        self.add_widget(self.limit_holder)
        self.add_widget(self.log_holder)
        self.add_widget(self.plot_button_holder)

    def CreateBindings(self):
        # Button Bindings
        self.data_fname_btn.bind(on_release=self.data_fname_dropdown.open)
        self.x_val_btn.bind(on_release=self.x_val_dropdown.open)
        self.y_val_btn.bind(on_release=self.y_val_dropdown.open)
        self.plot_btn.bind(on_release=self.PlotSelectedValues)
        self.clear_btn.bind(on_release=self.ClearPlot)
        self.save_btn.bind(on_release=self.SavePlot)

        # ToggleButton Bindings
        self.x_log_btn.bind(state=self.LogrithmicX)
        self.y_log_btn.bind(state=self.LogrithmicY)

        # Dropdown bindings
        self.data_fname_dropdown.bind(on_select=self.ProcessDataFile)
        self.x_val_dropdown.bind(on_select=lambda inst,x: setattr(self.x_val_btn,'text',x))
        self.y_val_dropdown.bind(on_select=lambda inst,x: setattr(self.y_val_btn,'text',x))

        # TextInput bindings
        self.x_min.bind(focus=self.UpdateAxisLimits)
        self.x_max.bind(focus=self.UpdateAxisLimits)
        self.y_min.bind(focus=self.UpdateAxisLimits)
        self.y_max.bind(focus=self.UpdateAxisLimits)
        self.x_label.bind(focus=self.UpdateXLabel)
        self.y_label.bind(focus=self.UpdateYLabel)
        self.title.bind(focus=self.UpdateTitle)

    def RebindDropdowns(self):
        self.x_val_btn.bind(on_release=self.x_val_dropdown.open)
        self.y_val_btn.bind(on_release=self.y_val_dropdown.open)
        self.x_val_dropdown.bind(on_select=lambda inst,x: setattr(self.x_val_btn,'text',x))
        self.y_val_dropdown.bind(on_select=lambda inst,x: setattr(self.y_val_btn,'text',x))

    # Callbacks
    def CreateDataFilenameDropdown(self,datadir):
        dropdown = DropDown()
        for (dirpath, dirnames, filenames) in os.walk(datadir):
            for filename in sorted(filenames):
                if filename.endswith('.csv'):
                    btn = Button(text=filename,size_hint_y=None,height=44)
                    btn.bind(on_release=lambda btn: dropdown.select(btn.text))
                    dropdown.add_widget(btn)
        return dropdown

    def CreateDataLabelDropdown(self,dropdown):
        for label in self.data.keys():
            btn = Button(text=label,size_hint_y=None,height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)
        return dropdown

    def ProcessDataFile(self,instance,x):
        self.x_val_dropdown.clear_widgets()
        self.y_val_dropdown.clear_widgets()
        setattr(self.data_fname_btn,'text',x)
        self.data = {}
        with open(self.datadir+x) as csvf:
            reader = csv.reader(csvf, delimiter = self.delim)
            for label in reader.__next__():
                self.data[label] = []
            n_label = len(self.data.keys())
            for row in reader:
                print(len(row), n_label)
                assert len(row)==n_label,'\n\tCSV Error: Number of data != Number of labels'
                i=0
                for label in self.data.keys():
                    if row[i]!='':
                        self.data[label].append(float(row[i]))
                    else:
                        self.data[label].append(0)
                    i+=1
        self.CreateDataLabelDropdown(self.x_val_dropdown)
        self.CreateDataLabelDropdown(self.y_val_dropdown)
        self.RebindDropdowns()

    # Plotting methods
    def PlotSelectedValues(self, instance):
        x_label = self.x_val_btn.text
        y_label = self.y_val_btn.text
        x_data = self.data[x_label]
        y_data = self.data[y_label]
        plt.scatter(x_data,y_data,s=5,label=y_label)
        #plt.legend()
        canvas = plt.gcf().canvas
        canvas.draw()
        x_min,x_max = plt.gca().get_xlim()
        y_min,y_max = plt.gca().get_ylim()
        self.x_min.text = str(x_min)
        self.x_max.text = str(x_max)
        self.y_min.text = str(y_min)
        self.y_max.text = str(y_max)

    def ClearPlot(self,instance):
        plt.clf()
        canvas = plt.gcf().canvas
        canvas.draw()

    def SavePlot(self,instance):
        if self.title != '':
            fname = self.plotdir+self.title.text+'.png'
        else:
            fname = self.plotdir+'default'
        fig = plt.figure()
        x_label = self.x_val_btn.text
        y_label = self.y_val_btn.text
        x_data = self.data[x_label]
        y_data = self.data[y_label]
        plt.scatter(x_data,y_data,s=5)
        plt.title(self.title.text)
        plt.xlabel(self.x_label.text)
        plt.ylabel(self.y_label.text)
        try:
            a = [float(self.x_min.text),float(self.x_max.text),float(self.y_min.text),float(self.y_max.text)]
        except ValueError:
            a=None
        if a!=None:
            plt.axis(a)
        fig.savefig(fname,bbox_inches='tight',format='png')
        plt.close(fig)

    def UpdateAxisLimits(self,inst,val):
        try:
            a = [float(self.x_min.text),float(self.x_max.text),float(self.y_min.text),float(self.y_max.text)]
        except ValueError:
            return
        if not val:
            plt.axis(a)
            canvas = plt.gcf().canvas
            canvas.draw()

    def UpdateXLabel(self,inst,val):
        if not val:
            plt.xlabel(self.x_label.text)
            canvas = plt.gcf().canvas
            canvas.draw()

    def UpdateYLabel(self,inst,val):
        if not val:
            plt.ylabel(self.y_label.text)
            canvas = plt.gcf().canvas
            canvas.draw()

    def UpdateTitle(self,inst,val):
        if not val:
            plt.title(self.title.text)
            canvas = plt.gcf().canvas
            canvas.draw()

    def LogrithmicX(self,inst,val):
        ax = plt.gca()
        if val=='down':
            ax.set_xscale('log')
        else:
            ax.set_xscale('linear')
        canvas = plt.gcf().canvas
        canvas.draw()

    def LogrithmicY(self,inst,val):
        ax = plt.gca()
        if val=='down':
            ax.set_yscale('log')
        else:
            ax.set_yscale('linear')
        canvas = plt.gcf().canvas
        canvas.draw()

class PlotApp(App):
    def build(self):
        Window.size=(1200,800)
        main_area = BoxLayout(orientation='horizontal',spacing=25,padding=10)
        fig = plt.figure()
        select_area = SelectionArea(size_hint_x=0.7)
        main_area.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        main_area.add_widget(select_area)
        return main_area

if __name__=='__main__': PlotApp().run()
