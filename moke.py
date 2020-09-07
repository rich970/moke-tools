#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:45:51 2020

@author: richard
"""

import numpy as np
import re
import matplotlib.pyplot as plt


class moketools(object):
    '''
    The moketools object contains various tools to assist the analysis of
    MOKE measurements.

    Methods:

        columns()
        addcolumn(col_data, col_name)
        importMOKEdata(filename)
        findmax(col)
        findmix(col)
        centreloop()
        centreloopnormalise()
        findHc()
        centrefield()
        findrem()

    Methods to add:
        fixdrift()
        delcolumn()

    Methods to improves:
        importMOKEdata(filename):generalize identification of where data starts
    '''

    def __init__(self, data, **kwargs):
        """ Initialises the Moke object from the datafile. Type: object """
        self.columns = kwargs.get('columns', None)
        self.header = kwargs.get('header', None)
        self.data = data

    def __len__(self):
        """ Returns the number of data points. Type: int """
        return len(self.data)

    def __str__(self):
        string = 'Moke object - Columns: {0} \nData: \n{1}'.format(
             self.columns, self.data[0:4])
        return string

    def columns(self):
        """ Method to return the column names. Type: string """
        return self.columns

    def addcolumn(self, col_data, col_name, data_type=float):
        """ Method to add a column.
        Parameters
        ----------
        col_data : list
            A list of data to be added. Type can be string, int or float.
        col_name : string
            The name of the column which should be used as a reference.
            col_name must be unique.
        data_type : The type of data included in col_data.

        Returns
        -------
        None
        """
        # Because the array is structured, we have to create a new array
        # First append the data-type for the new column
        new_dtype = self.data.dtype.descr + [(col_name, data_type)]
        data = np.empty(shape=len(self.data['index']),
                        dtype=new_dtype)
        # Use for loop to copy over the old data in the new structured array
        for col in self.data.dtype.names:
            data[col] = self.data[col]
        # Finally, add the new column to the new structured array.
        data[col_name] = col_data
        self.data = data  # Overwrite the data field in the moketools object
        self.columns = data.dtype.names

    def delcolumn(self, col_name):
        """ Method to delete a column.
        Parameters
        ----------
        col_name : string
            The name of the column to be deleted
        data_type : The type of data included in col_data.

        Returns
        -------
        None
        """
        names = list(self.data.dtype.names)  # list of column names
        try:
            ind = names.index(col_name)  # index of column to be deleted
        except ValueError:  # if the column does exist...
            print('Error: \'{0}\' is not a column'.format(col_name))
            return

        # Copy the old data types then remove the one for the deleted column
        new_dtype = list(self.data.dtype.descr)
        new_dtype.remove((new_dtype[ind][0], new_dtype[ind][1]))

        # Create a new data area and populate with the columns we want to keep
        data = np.empty(shape=len(self.data['index']),
                        dtype=new_dtype)
        for col in self.data.dtype.names:
            if col != col_name:
                data[col] = self.data[col]
        self.data = data  # Overwrite the data field in the moketools object
        self.columns = data.dtype.names

    def head(self, n=5):
        """ Prints the columns name and the first few lines of data.
        Parameters: n : int
            Number of data lines to print
        Returns: None
        """

        print(self.data.dtype.names)
        for dataline in self.data[:n]:  # Print the first few lines of data
            print(dataline)

    def importMOKEdata(filename, **kwargs):
        """
        Given a filename/filepath this will import the data, extract the
        headers and column names and return a Moke object

        Parameters
        ----------
        filename : string
            filepath or filename of the data to be imported..

        Returns
        -------
        TYPE Moke object
            Moke object with structured data, columns and headers.

        """
        ld = open(filename, 'r')
        string = ld.read()
        # Split the data string each time '***End_of_Header***'
        regex = re.split(r'\*\*\*End_of_Header\*\*\*', string)
        delim = kwargs.get('delim', '\t')
        header = ''
        columns = list
        index = []
        field = []
        kerrV = []

        # Merge each 'End_of_Header' split, excluding last which is the data
        for split in regex[:-1]:
            header = header + split
        print(header)
        # regex[-1] will contain just data. Split this into each new line.
        regex = re.split(r'\n', regex[-1])
        # For these Labview files regex[0] now contains 4x delimeter markers
        #                        regex[1] contains the column labels
        #                       regex[2:end] contains the data
        # FIX - Something more generalised should be implemented at some point
        columns = re.split(delim, regex[1])
        # These column names in the file are meaningless!
        print('Column names from file:\n {0}'.format(columns))

        # find number of data columns using delimeter provided
        n_data_cols = len(re.findall(delim, regex[2]))
        # Find the numerical data line by line
        count = 0
        for line in regex:
            dataline = np.array(re.findall('.\d*\.\d*', line))
            # only use datalines which have the right number of columns
            if np.size(dataline) == n_data_cols:
             #   index.append(int(float(dataline[0])))
                index.append(count)
                field.append(float(dataline[1]))
                kerrV.append(float(dataline[2]))
                count += 1

        data = np.empty(shape=(len(index)),
                        dtype=[('index', int),
                               ('field', float),
                               ('kerrV', float)])
        data['index'] = index
        data['field'] = field
        data['kerrV'] = kerrV
        print('Assuming columns name are: \n{0}'.format(data.dtype.names))
        columns = data.dtype.names  # redefine columns variable
        for dataline in data[:5]:  # Print the first few lines of data
            print(dataline)
        return moketools(data=data, columns=columns, header=header)

    def findmax(self, col):
        """
        Finds index of the absolute maximum. Performs an average of +- 5
        datapoints from this index and returns that as the max value.

        Parameters
        ----------
        col : TYPE int
            Datacolumn for which the maximum should be found. Typically the
            Kerr voltage.

        Returns
        -------
        mx : TYPE float
            The maximum value, averaged over the neighbouring datapoints from
            the absolute maximum value.
        """
        ind = np.argmax(self.data[col])
        mx = np.average(self.data[col][ind-5:ind+5])
        return mx

    def findmin(self, col):
        """
        Finds index of the absolute minimum. Performs an average of +- 5
        datapoints from this index and returns that as the min value.

        Parameters
        ----------
        col : TYPE int
            Datacolumn for which the maximum should be found. Typically the
            Kerr voltage.

        Returns
        -------
        mn : TYPE float
            The minimum value, averaged over the neighbouring datapoints from
            the absolute minimum value.
        """
        ind = np.argmin(self.data[col])
        mn = np.average(self.data[col][ind-5:ind+5])
        return mn

    def normalise(self, col):
        """ Nomalises a column. Creates a new column with name of the format
        'normalised-[col]'
        Parameters
        ----------
            col : string
            Name of the col to be normalised.
        """

        mx = self.findmax(col)
        normalised = self.data[col]/mx
        moketools.addcolumn(self, normalised, 'normalised-{0}'.format(col))

    def centreloop(self):
        """ Centers the MOKE data about zero. Creates a new column with name
        'centered-Kerr'
        """
        mx = self.findmax('kerrV')
        mn = self.findmin('kerrV')
        centered = (self.data['kerrV'] - 0.5*(mx + mn))
        moketools.addcolumn(self, centered, 'centered-Kerr')

    def centreloopnormalise(self):
        """ Centers the MOKE data about zero and normalises it between +- 1.
        Creates a new column with name 'normalised-Kerr'
        """
        if 'centered-Kerr' not in self.data.dtype.names:
            self.centreloop()
        mx = self.findmax('centered-Kerr')
        normalised = self.data['centered-Kerr']/mx
        moketools.addcolumn(self, normalised, 'normalised-Kerr')

    def findHc(self):
        """ Finds the coercive field using the 'normalised-Kerr' column.
        Returns
        ----------
            Hc : list
            A size two list with a float type value for the Hc calculated
            from an up-sweep and a down-sweep, respectively.
        """
        if 'normalised-Kerr' not in self.data.dtype.names:
            return print('Loop must be centred and normalised first')
        L = len(self.data['index'])  # How many datapoint are there in this loop
        y = np.abs(self.data['normalised-Kerr'])
        i_upsweep = np.argmin(y[:int(L/2)])
        i_downsweep = np.argmin(y[int(L/2):])
        Hc = self.data['field'][[i_upsweep, int(L/2)+i_downsweep]]
        return Hc

    def centrefield(self):
        """ Centers the field axis such that the Hc calculated for up-sweep
        and down-sweeps are equal. It overwrites the original field data'
        """
        Hc = self.findHc()
        print('Using Hc boundaries: {0} to symmetrise field axis'.format(Hc))
        delta = 0.5*(Hc[0] + Hc[1])
        self.data['field'] = self.data['field'] - delta

    def findrem(self, minfield=10.0):
        """ Find the remenent magnestisation as a decimal between 0 and 1
        using the 'normalised-Kerr' column.
        Parameters
        ----------
            minfield : float
                Defines the field region over which the remenet magnesation
                is averaged over. e.g. minfield=10 will average over all 
                data points measured at less than 10 Oe. For clean data,
                this number should be as close to zero as possible.
        Returns
        _______
            Rem : float
                The remenet magnetisation.
        """
        if 'normalised-Kerr' not in self.data.dtype.names:
            return print('Loop must be centred and normalised first')
        # return all indices where field is close to zero
        ind = np.argwhere(abs(self.data['field']) < minfield)
        # remanence is the average normalised-Kerr in this low field region
        Rem = np.mean(abs(self.data['normalised-Kerr'][ind]))
        return Rem

    def rolling_err(self, window_size):
        """ Calculates the standard error across a data window. This data window
            shifts along the dataset to build a vector of errors, sampling the 
            noise in the time series. In order to characterise the noise correctly
            the window should be small enough that the data sample it contains is
            insensitive to the longer timescale signal variation associated with
            the field sweep.
            
            Parameters
            ----------
                window_size : int
                    Size of data sampling window for each rolling error calcuation
            Returns
            _______
                np.mean(err) : The average error from all the data samples.
        """
        err = []
        for ind in self.data['index']:
            # FIX - should implent cyclic iterations
            if ind+window_size < len(self.data['index']):
                s_window = range(ind, ind+window_size)
                std = np.std(self.data['corrected'][s_window])
                err.append(std/np.sqrt(window_size))
        return np.mean(err)

    def fixdrift(self, b=3, **kwargs):
        data_window = kwargs.get('data_window', 10)
        d_step = kwargs.get('d_step', 2)
        plot_figs = kwargs.get('plot_figs', False)

        # Create new column with the kerrV data to be corrected
        self.addcolumn(self.data['kerrV'], 'corrected')

        n = len(self.data['index'])
        mn = np.mean(self.data['kerrV'])
        split = round(n/2)  # Middle index of the dataset

        # Estimate the noise level on the data
        err = np.mean(moketools.rolling_err(self, data_window))
        # Initialise the param 'mean absolute error'for assessing fit
        mae = b*err*10
        # Loop over and fit until the mae is comparable to the noise level
        loop_n = 1
        print('Attempting to offset switched region to obtain optimal fit')
        if plot_figs:
            plt.figure()

        while mae/err > b:
            # Fit a 2nd order polynomial to the data for use later
            drift_params = np.polyfit(self.data['index'],
                                      self.data['corrected'], 2)
            model = drift_params[2] \
                + drift_params[1]*self.data['index'] \
                + drift_params[0]*self.data['index']**2

            # Calculate the mae with this model:
            mae = sum(abs(model-self.data['corrected'])
                      )/len(self.data['index'])

            # Now use the mae to try to compensate the switched region
            # Find indices of the magnetisation reversal
            diff = np.diff(self.data['corrected'])
            i1 = abs(diff)[:split].argmax()
            i2 = split + abs(diff)[split:].argmax()
            self.data['corrected'][i1:i2] = (self.data['corrected'][i1:i2]
                                             + np.sign(diff[i2])*mae*d_step)
            if plot_figs:
                plt.clf()
                plt.title('Attempting to compensate switched region')
                plt.xlabel('index')
                plt.ylabel('kerr voltage')
                # Plot the data so far
                plt.plot(self.data['index'], self.data['corrected'])
                # Where the switches were found:
                plt.plot(self.data['index'][i1-1], self.data['corrected'][i1-1], 'ko')
                plt.plot(self.data['index'][i2-1], self.data['corrected'][i2-1], 'ko')
                # MAE margin lines
                plt.plot(np.linspace(0, n, 10), mn*np.ones(10), 'k-')
                plt.plot(np.linspace(0, n, 10), mn-mae*np.ones(10), 'k--')
                plt.plot(np.linspace(0, n, 10), mn+mae*np.ones(10), 'k--',
                         label='mae')
                # The model...
                plt.plot(self.data['index'], model, '--b')
    
                # How does this compare to the standard error on the data?
                plt.plot(np.linspace(0, n, 10), mn*np.ones(10), '-')
                plt.plot(np.linspace(0, n, 10), mn+b*err*np.ones(10), 'r--')
                plt.plot(np.linspace(0, n, 10), mn-b*err*np.ones(10), 'r--',
                         label='noise-level')
                plt.legend()
                plt.pause(0.5)

            print('{0}: [MAE/noise level] = {1}'.format(loop_n, mae/err))
            loop_n += 1

            if loop_n > 20:
                print('Unable to correct drift. Consider lowering the \
                      threshold parameter (b)')
                return
        # Once the fit is decent,  we substract the drift model
        print('Success!')
        self.data['corrected'] = self.data['kerrV'] - model
        plt.figure()
        plt.plot(self.data['field'], self.data['corrected'])

class wafermap(object):
    '''
    The wafermap object contains various tools which allow spatial mapping
    across a wafer of magnetic parameters extracted through MOKE measurements

    Methods:
        __init__(waferdiamter=float, divsize=float)
        draw_circle(ax, alpha=0.6, color='w--')
        def draw_wafermap(ax, origin=[lower, upper], title=string):
        def label_params(comp,divs,ax):
    '''

    def __init__(self, waferdiameter, divsize):
        """ Initialises the wafermap object. Type: object """
        ndivs = int(np.ceil(waferdiameter/divsize))
        self.waferdiameter = waferdiameter
        self.divsize = divsize
        self.waferarray = np.zeros(shape=(ndivs, ndivs))

    def label_param(self, pos, param):
        """ Assigns a parameter to a position in the wafer array """
        self.waferarray[pos[0], pos[1]] = param

    def draw_wafermap(self, ax, **kwargs):

        origin = kwargs.get('origin', 'lower')
        title = kwargs.get('title', 'Wafer parameter map')
        ndivs = int(np.ceil(self.waferdiameter/self.divsize))
        edge = self.divsize*ndivs
        q = np.linspace(-edge, edge, ndivs+1)+self.divsize

        ax.imshow(self.waferarray, interpolation='none', 
                  origin=origin)
        ax.set_xticks(np.arange(len(q)))
        ax.set_yticks(np.arange(len(q)))
        # round label to 2.d.p and change orientation
        ax.set_xticklabels(np.round(q, 2),rotation=-45)
        ax.set_yticklabels(np.round(q, 2))
        ax.set_xlabel('x-distance from magnetron focus [mm]')
        ax.set_ylabel('y-distance from magnetron focus [mm]')
        ax.set_title(title)
        ax.set_xlim(0, ndivs-1)
        #ax.set_ylim(0, ndivs-1)

        #make a grid using the minor ticks as reference
        ax.set_xticks(np.arange(len(q))-0.5, minor=True)
        ax.set_yticks(np.arange(len(q))-0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)

    def draw_circle(self, ax, alpha=0.6, color='w--'):
        """
        Draws a circle on the figure

        Parameters
        ----------
        radius : float
            Radius of the circle to be draw.
        divs : float
            Size of the divisions for which the wafer is divided.
        n_divs : int
            Number of division for which the wafer is divided.
        ax : axes handle
            Handle of the axes where the circle will be drawn
    
        Returns
        -------
        None.

        """
        ndivs = int(np.ceil(self.waferdiameter/self.divsize))
        cen = (ndivs-1)/2
        x = cen+((self.waferdiameter/2)/self.divsize)*np.cos(np.linspace(-np.pi,np.pi,100))
        y = cen+((self.waferdiameter/2)/self.divsize)*np.sin(np.linspace(-np.pi,np.pi,100))   
        ax.plot(x,y,color, linewidth = 2, alpha = alpha)


