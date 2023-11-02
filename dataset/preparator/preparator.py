import os
import numpy as np
import h5py
import scipy.io
import pandas as pd
import re
from tqdm import tqdm
import os



class Preparator:
    def __init__(self, load_directory='./', save_directory='./', load_file_pattern='*', save_file_name='all.npz',verbose=False):

        self.load_directory = load_directory
        self.save_directory = save_directory
        self.load_file_pattern = re.compile(load_file_pattern)
        self.save_file_name = save_file_name
        self.extract_pattern = None
        self.extract_pattern = None
        self.start_time = None
        self.length_time = None
        self.start_channel = None
        self.end_channel = None
        self.on_blocks = None
        self.off_blocks = None
        self.filters = []
        self.ignore_events = []
        self.labels = []
        self.verbose = verbose
        self.padding = True
        print("Preparator is initialized with: ")
        print("Directory to load data: " + self.load_directory)
        print("Directory to save data: " + self.save_directory)
        print("Looking for file that match: " + load_file_pattern)
        print("Will store the merged file with name: " + self.save_file_name)


    def extract_data_at_events(self, extract_pattern, name_start_time, start_time, name_length_time, length_time, start_channel, end_channel, padding=True):
        self.extract_pattern = extract_pattern
        self.start_time = start_time
        self.length_time = length_time
        self.start_channel = start_channel
        self.end_channel = end_channel
        self.padding = padding

        print("Preparator is instructed to look for events that match structure: " + str(self.extract_pattern))
        print("Time dimension -- Cut start info: " + name_start_time)
        print("Time dimension -- Cut length info: " + name_length_time)
        print("Channel dimension -- Cut start info: " + str(start_channel))
        print("Channel dimension -- Cut end info: " + str(end_channel))

    def blocks(self, on_blocks, off_blocks):
        self.on_blocks = on_blocks
        self.off_blocks = off_blocks
        print("Blocks to be used are: " + str(on_blocks))
        print("Blocks to be ignored are: " + str(off_blocks))

    def addFilter(self, name, f):
        self.filters.append((name, f))
        print('Preparator is instructed to use filter: ' + name)

    def addLabel(self, name, f):
        self.labels.append((name, f))
        print('Preparator is instructed to use label: ' + name)

    def ignoreEvent(self, name, f):
        self.ignore_events.append((name, f))
        print('Preparator is instructed to ignore the event: ' + name)

    def run(self):
        print("Starting collecting data.")
        all_EEG = []
        all_labels = []
        subj_counter = 1


        progress = tqdm(sorted(os.listdir(self.load_directory)))
        for subject in progress:

            if os.path.isdir(self.load_directory + subject):
                # if subject == 'BY2':
                #    continue
                # if subject == 'EP18':
                #    break

                cur_dir = self.load_directory + subject + '/'
                for f in sorted(os.listdir(cur_dir)):
                    if not self.load_file_pattern.match(f):
                        continue

                    progress.set_description('Loading ' + f)
                    # load the mat file
                    events = None
                    data = None
                    # preparator.py - line 93
                    if h5py.is_hdf5(cur_dir + f):
                        hdf5file = h5py.File(cur_dir + f, 'r')
                        EEG = hdf5file[list(hdf5file.keys())[1]]  # removal of a repeated h5py.File() call here
                        events = self._load_hdf5_events(EEG)
                        data = np.array(EEG['data'], dtype='float')
                    else:
                        matfile = scipy.io.loadmat(cur_dir + f)
                        EEG = matfile[list(matfile.keys())[3]][0,0] #eventually at the end to remove the repetition in the load method
                        events = self._load_v5_events(EEG)
                        data = np.array(EEG['data'], dtype='float').T


                    events = self._ignore_events(events)
                    if self.verbose: print(events)
                    select = self._filter_blocks(events)
                    select &= self._filter_events(events)
                    trials = self._extract_events(data, events, select)
                    labels = self._extract_labels(events, select, subj_counter)

                    all_EEG.append(trials)
                    all_labels.append(labels)
                subj_counter += 1

        # save the concatenated arrays
        print('Saving data...')
        EEG = np.concatenate(all_EEG, axis=0, dtype='float')
        labels = np.concatenate(all_labels, axis=0, dtype='float')
        print("Shapes of EEG are: ")
        print(EEG.shape)
        print("Shapes of labels are: ")
        print(labels.shape)
        np.savez(self.save_directory + self.save_file_name, EEG=EEG, labels=labels)


    def _load_v5_events(self, EEG):
        if self.verbose: print("Loading the events from the subject. ")
        # extract the useful event data
        events = pd.DataFrame()
        events['type'] = [el[0].strip() for el in EEG['event'][0]['type']]
        # if self.verbose: print(events)
        events['latency'] = [el[0, 0] for el in EEG['event'][0]['latency']]
        # if self.verbose: print(events)
        events['amplitude'] = [el[0, 0] for el in EEG['event'][0]['sac_amplitude']]
        # if self.verbose: print(events)
        events['start_x'] = [el[0, 0] for el in EEG['event'][0]['sac_startpos_x']]
        # if self.verbose: print(events)
        events['end_x'] = [el[0, 0] for el in EEG['event'][0]['sac_endpos_x']]
        # if self.verbose: print(events)
        events['start_y'] = [el[0, 0] for el in EEG['event'][0]['sac_startpos_y']]
        # if self.verbose: print(events)
        events['end_y'] = [el[0, 0] for el in EEG['event'][0]['sac_endpos_y']]
        # if self.verbose: print(events)
        events['duration'] = [el[0, 0] for el in EEG['event'][0]['duration']]
        # if self.verbose: print(events)
        events['avgpos_x'] = [el[0, 0] for el in EEG['event'][0]['fix_avgpos_x']]
        # if self.verbose: print(events)
        events['avgpos_y'] = [el[0, 0] for el in EEG['event'][0]['fix_avgpos_y']]
        # if self.verbose: print(events)
        events['endtime'] = [el[0, 0] for el in EEG['event'][0]['endtime']]
        
        events['avgpupilsize'] = [el[0, 0] for el in EEG['event'][0]['fix_avgpupilsize']]
        

        if self.verbose:
            print("Events loaded are: ")
            print(events)
        return events

    def _load_hdf5_events(self, EEG):
        if self.verbose: print("Loading the events from the subject. ")
        # extract the useful event data
        events = pd.DataFrame()
        events['type'] = [''.join(map(chr, EEG[ref][:, 0])).strip() for ref in EEG['event']['type'][:, 0]]
        #if self.verbose: print(events)
        events['latency'] = [EEG[ref][0, 0] for ref in EEG['event']['latency'][:, 0]]
       # if self.verbose: print(events)
        events['amplitude'] = [EEG[ref][0, 0] for ref in EEG['event']['sac_amplitude'][:, 0]]
        #if self.verbose: print(events)
        events['start_x'] = [EEG[ref][0, 0] for ref in EEG['event']['sac_startpos_x'][:, 0]]
        #if self.verbose: print(events)
        events['end_x'] = [EEG[ref][0, 0] for ref in EEG['event']['sac_endpos_x'][:, 0]]
        #if self.verbose: print(events)
        events['start_y'] = [EEG[ref][0, 0] for ref in EEG['event']['sac_startpos_y'][:, 0]]
        #if self.verbose: print(events)
        events['end_y'] = [EEG[ref][0, 0] for ref in EEG['event']['sac_endpos_y'][:, 0]]
        #if self.verbose: print(events)
        events['duration'] = [EEG[ref][0, 0] for ref in EEG['event']['duration'][:, 0]]
        #if self.verbose: print(events)
        events['avgpos_x'] = [EEG[ref][0, 0] for ref in EEG['event']['fix_avgpos_x'][:, 0]]
        #if self.verbose: print(events)
        events['avgpos_y'] = [EEG[ref][0, 0] for ref in EEG['event']['fix_avgpos_y'][:, 0]]
        # if self.verbose: print(events)
        events['endtime'] = [EEG[ref][0, 0] for ref in EEG['event']['endtime'][:, 0]]
        
        events['avgpupilsize'] = [EEG[ref][0, 0] for ref in EEG['event']['fix_avgpupilsize'][:, 0]]

        if self.verbose:
            print("Events loaded are: ")
            print(events)

        return events

    def _filter_blocks(self, events):
        if self.verbose: print("Filtering the blocks: ")
        select = events['type'].apply(lambda x: True)

        if self.on_blocks is None or self.off_blocks is None:
            return select

        for i, event in enumerate(events['type']):
            if event in self.on_blocks:
                select.iloc[i] = True
            elif event in self.off_blocks:
                select.iloc[i] = False
            elif i > 0:
                select.iloc[i] = select.iloc[i-1]
        if self.verbose: print(list(zip(range(1, len(select) + 1), select)))
        return select

    def _ignore_events(self, events):
        ignore = events['type'].apply(lambda x: False)
        for name, f in self.ignore_events:
            if self.verbose: print("Applying: " + name)
            ignore |= f(events)
            if self.verbose: print(list(zip(range(1, len(ignore) + 1), ignore)))
        select = ignore.apply(lambda x: not x)
        if self.verbose: print(list(zip(range(1, len(select) + 1), select)))
        return events.loc[select]

    def _filter_events(self, events):
        if self.verbose: print("Filtering the events: ")
        select = events['type'].apply(lambda x: True)
        for i, event in enumerate(self.extract_pattern):
            select &= events['type'].shift(-i).isin(event)
        if self.verbose: print(list(zip(range(1, len(select) + 1), select)))

        for name, f in self.filters:
            if self.verbose: print("Applying filter: " + name)
            select &= f(events)
            if self.verbose: print(list(zip(range(1, len(select) + 1), select)))
        return select

    def _extract_events(self, data, events, select): # needs to be able to pad
        if self.verbose: print("Extracting data from the interested events: ")

        all_trials = []
        # extract the useful data
        if self.verbose: print(data)

        start = self.start_time(events).loc[select]
        length = events['type'].apply(lambda x: self.length_time).loc[select]
        end_block = events['latency'].shift(-len(self.extract_pattern)).loc[select]
        if self.verbose: print(start)
        if self.verbose: print(length)
        if self.verbose: print(end_block)
        for s, l, e in zip(start, length, end_block):
            assert(e > s)
            if s + l > e and self.padding:
                #Need to pad since, the required length is bigger then the last block
                if self.verbose: print(str(s) + ", " + str(l) + ", " + str(e) + " that is need to pad")
                unpaded_data = data[int(s - 1):int(e - 1), (self.start_channel - 1):self.end_channel]
                padding_size = int(s + l - e)
                append_data = np.pad(unpaded_data, pad_width=((0, padding_size), (0, 0)), mode='reflect')
                if self.verbose: print(append_data)
            else:
                append_data = data[int(s - 1):int(s + l - 1), (self.start_channel - 1):self.end_channel]

            all_trials.append(append_data)

        all_trials = np.array(all_trials)
        # if self.verbose: print("Extracted all this data from this participant.")
        # if self.verbose: print(all_trials)
        return all_trials

    def _extract_labels(self, events, select, subj_counter):
        if self.verbose: print("Extracting the labels for each trial.")
        if self.verbose: print("Appending the subject counter. ")
        # append subject IDs to the full list and then the labels
        nr_trials = events.loc[select].shape[0]
        labels = np.full((nr_trials, 1), subj_counter, dtype='float')
        if self.verbose: print(labels)

        for name, f in self.labels:
            if self.verbose: print("Appending the next label: " + name)
            labels = np.concatenate((labels, np.asarray(f(events).loc[select], dtype='float').reshape(-1,1)), axis=1)
            if self.verbose: print(labels)
        return labels
