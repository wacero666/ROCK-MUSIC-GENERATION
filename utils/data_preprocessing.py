#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 12:06:46 2018

Taken from Hedonistr's Music Generation With Lstm blog post

"""

import zipfile
import glob

from pathlib import Path

import glob
import os
import music21
import math
import pprint as pp
import numpy as np
import scipy as sp
import scipy.io as sio
import matplotlib.pyplot as plt
from music21 import converter, instrument, note, chord, duration, stream
print (music21.__version__) #if your version is lower than 4.x.x, you will encounter with some issues.

def note_to_int(note): # converts the note's letter to pitch value which is integer form.
    # source: https://musescore.org/en/plugin-development/note-pitch-values
    # idea: https://github.com/bspaans/python-mingus/blob/master/mingus/core/notes.py
    note_base_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if ('#-' in note):
        first_letter = note[0]
        base_value = note_base_name.index(first_letter)
        octave = note[3]
        value = base_value + 12*(int(octave)-(-1))
        
    elif ('#' in note): # not totally sure, source: http://www.pianofinders.com/educational/WhatToCallTheKeys1.htm
        first_letter = note[0]
        base_value = note_base_name.index(first_letter)
        octave = note[2]
        value = base_value + 12*(int(octave)-(-1))
        
    elif ('-' in note): 
        first_letter = note[0]
        base_value = note_base_name.index(first_letter)
        octave = note[2]
        value = base_value + 12*(int(octave)-(-1))
        
    else:
        first_letter = note[0]
        base_val = note_base_name.index(first_letter)
        octave = note[1]
        value = base_val + 12*(int(octave)-(-1))
        
    return value


# Lets determine our matrix's value 
# rest --> (min_value, lower_bound)
# continuation --> (lower_bound, upper_bound)
# first_touch --> (upper_bound, max_value)

min_value = 0.00
lower_first = 0.00

lower_second = 0.5
upper_first = 0.5

upper_second = 1.0
max_value = 1.0

def notes_to_matrix(notes, durations, offsets, min_value=min_value, lower_first=lower_first,
                    lower_second=lower_second,
                    upper_first=upper_first, upper_second=upper_second,
                    max_value=max_value):
    
    # I want to represent my notes in matrix form. X axis will represent time, Y axis will represent pitch values.
    # I should normalize my matrix between 0 and 1.
    # So that I will represent rest with (min_value, lower_first), continuation with [lower_second, upper_first]
    # and first touch with (upper_second, max_value)
    # First touch means that you press the note and it cause to 1 time duration playing. Continuation
    # represent the continuum of this note playing. 
    
    try:
        last_offset = int(offsets[-1]) 
    except IndexError:
        print ('Index Error')
        return (None, None, None)
    
    total_offset_axis = last_offset * 4 + (8 * 4) + 1000 # +16 is adam hack...
    our_matrix = np.random.uniform(min_value, lower_first, (128, int(total_offset_axis))) 
    # creates matrix and fills with (-1, -0.3), this values will represent the rest.
    
    for (note, duration, offset) in zip(notes, durations, offsets):
        how_many = int(float(duration)/0.25) # indicates time duration for single note.
       
        
        # Define difference between single and double note.
        # I have choose the value for first touch, the another value for contiunation
        # lets make it randomize
        first_touch = np.random.uniform(upper_second, max_value, 1)
        # continuation = np.random.randint(low=-1, high=1) * np.random.uniform(lower_second, upper_first, 1)
        continuation = np.random.uniform(lower_second, upper_first, 1)
        if ('.' not in str(note)): # it is not chord. Single note.
            our_matrix[note, int(offset * 4)] = first_touch
            our_matrix[note, int((offset * 4) + 1) : int((offset * 4) + how_many)] = continuation

        else: # For chord
            chord_notes_str = [note for note in note.split('.')] 
            chord_notes_float = list(map(int, chord_notes_str)) # take notes in chord one by one

            for chord_note_float in chord_notes_float:
                our_matrix[chord_note_float, int(offset * 4)] = first_touch
                our_matrix[chord_note_float, int((offset * 4) + 1) : int((offset * 4) + how_many)] = continuation
                
    return our_matrix


def check_float(duration): # this function fix the issue which comes from some note's duration. 
                           # For instance some note has duration like 14/3 or 7/3. 
    if ('/' in duration):
        numerator = float(duration.split('/')[0])
        denominator = float(duration.split('/')[1])
        duration = str(float(numerator/denominator))
    return duration

def get_instrument_by_id(id):
    switcher = {
        0: "Piano",
        1: "N/A", # Chromatic Percussion
        2: "Piano", # Organ
        3: "Guitar",
        4: "Bass",
        5: "N/A", # Strings
        6: "N/A", # Ensemble
        7: "N/A", # Brass
        8: "N/A", # Reed
        9: "N/A", # Pipe
        10: "Piano", # Synth Lead
        11: "Piano", # Synth Pad
        12: "N/A",   # Synth Effects
        13: "N/A",   # Ethnic
        14: "N/A",   # Percussive
        15: "N/A",   # Sound Effects
    }
    return switcher[math.floor(id/8)]

def extract_instrument_labels(filepath):
    # Track labels should be stored in the midiProgram data of parsed streams, 
    # though for some odd reason music21 misreads this a lot... instead, we can
    # work around it by looking at program changes in the MIDI file events
    #
    # Outputs are trackChannels, trackInstrumentIds, and emptyTrackIndices

    # parse the given MIDI file
    mf = music21.midi.MidiFile()  
    mf.open(filepath)
    mf.read() 
    mf.close()

    # Look at each track and extract relevant info
    midiTracks = mf.tracks
    trackChannels = []      # In the given MIDI file, what channel # the track is
    trackInstrumentIds = [] # Ids for what instrument is on the given track
    emptyTrackIndices = []  # Tracks that have no musical content (notes or lyrics)
    count = 0
    
    for t in midiTracks:
        if t.hasNotes():
            trackChannels.append(t.getChannels()[1])
            if t.getProgramChanges():
                trackInstrumentIds.append(t.getProgramChanges()[0])
            else: # no program cahnge; arbitrarily assign piano for now...
                trackInstrumentIds.append(0) 
        else:
            trackChannels.append(-1)
            trackInstrumentIds.append(-1)
            emptyTrackIndices.append(count)
            count += 1;

    # map each to relevant instrument names
    instrumentNames = []
    for i in list(range(len(trackChannels))):
        if trackChannels[i] == -1:
            continue
        elif trackChannels[i] == 10:
            instrumentNames.append("Drums")
        else:
            instrumentNames.append(get_instrument_by_id(trackInstrumentIds[i]))
            
    return instrumentNames



def midi_to_matrix(filename, length=250, wantedInstruments = ['Guitar','Bass','Drums']): # convert midi file to matrix for DL architecture.
    
    mididata = music21.converter.parse(filename)
    
    # get instrument names    
    instrument_names = extract_instrument_labels(filename)
    print('Instruments found: ')
    print(instrument_names)

    
    full_output = [[[0 for k in range(length)] for j in range(128)] for i in range(len(wantedInstruments))]
    
    for d in list(range(len(wantedInstruments))):
        if wantedInstruments[d] in instrument_names:
            # pick whichever track has the instrument and has the most notes/info
            indices = [i for i, x in enumerate(instrument_names) if x == wantedInstruments[d]] # finds all indices of desired instrument
            notes_to_parse = []
            # indicesIndex = 0
            for i in indices: 
                temp_notes_to_parse = mididata.parts[i].recurse();
                if len(temp_notes_to_parse) > len(notes_to_parse):
                    notes_to_parse = temp_notes_to_parse
                    # indicesIndex = i;
            
            # Extract the relevant information from the track
            durations = [];
            notes = [];
            offsets = [];
            
            for element in notes_to_parse:
                if isinstance(element, note.Note): # if it is single note
                    notes.append(note_to_int(str(element.pitch)))
                    duration = str(element.duration)[27:-1]
                    durations.append(check_float(duration))
                    offsets.append(element.offset)
                elif isinstance(element, chord.Chord): # if it is chord
                    notes.append('.'.join(str(note_to_int(str(n)))
                                          for n in element.pitches))
                    duration = str(element.duration)[27:-1]
                    durations.append(check_float(duration))
                    offsets.append(element.offset)
                #else:
                    #print(element) # probably just rests and other metadata
                    #print ('No %s part in %s' %(wantedInstruments[d], filename))
                    
            current_matrix = notes_to_matrix(notes, durations, offsets)
            try:
                freq, time = current_matrix.shape
            except AttributeError:
                print ("'tuple' object has no attribute 'shape'")
                return None
            
            if (time >= length):
                current_output = (current_matrix[:,:length]) # We have to set all individual note matrix to same shape for Generative DL.
            else:
                print ('%s is not long enough' %(filename))
            
            for i in range(128):
                for j in range(length):
                    full_output[d][i][j] = current_output[i][j]
    return full_output
        

        
def int_to_note(integer):
    # convert pitch value to the note which is a letter form. 
    note_base_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave_detector = (integer // 12) 
    base_name_detector = (integer % 12) 
    note = note_base_name[base_name_detector] + str((int(octave_detector))-1)
    if ('-' in note):
      note = note_base_name[base_name_detector] + str(0)
      return note
    return note

# ATTENTION. From matrix form to midi form, I have to indicate first touch, continuation and rest with unique numbers.
# I choose -1.0 for rest , 0 for continuation and 1 for first touch.

lower_bound = (lower_first + lower_second) / 2
upper_bound = (upper_first + upper_second) / 2



def converter_func(arr,first_touch = 1.0, continuation = 0.0, lower_bound = lower_bound, upper_bound = upper_bound):
    # I can write this function thanks to https://stackoverflow.com/questions/16343752/numpy-where-function-multiple-conditions
    # first touch represent start for note, continuation represent continuation for first touch, 0 represent end or rest
    np.place(arr, arr < lower_bound, -1.0)
    np.place(arr, (lower_bound <= arr) & (arr < upper_bound), 0.0)
    np.place(arr, arr >= upper_bound, 1.0)
    return arr

def how_many_repetitive_func(array, from_where=0, continuation=0.0):
    new_array = array[from_where:]
    count_repetitive = 1 
    for i in new_array:
        if (i != continuation):
            return (count_repetitive)
        else:
            count_repetitive += 1
    return (count_repetitive)

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    


def matrix_to_midi(matrix, instName):
    first_touch = 1.0
    continuation = 0.0
    y_axis, x_axis = matrix.shape
    output_notes = []
    offset = 0
            
    matrix = matrix.astype(float)
    
    print (y_axis, x_axis)  # ADAM YOU'RE HERE debugging why the output fails

    for y_axis_num in range(y_axis):
        one_freq_interval = matrix[y_axis_num,:] # get a column
        # freq_val = 0 # columdaki hangi rowa baktığımızı akılda tutmak için
        one_freq_interval_norm = converter_func(one_freq_interval)
        # print (one_freq_interval)
        i = 0        
        offset = 0
        while (i < len(one_freq_interval)):
            how_many_repetitive = 0
            temp_i = i
            if (one_freq_interval_norm[i] == first_touch):
                how_many_repetitive = how_many_repetitive_func(one_freq_interval_norm, from_where=i+1, continuation=continuation)
                i += how_many_repetitive 

            if (how_many_repetitive > 0):
                new_note = note.Note(int_to_note(y_axis_num),duration=duration.Duration(0.25*how_many_repetitive))
                new_note.offset = 0.25*temp_i
                if instName is "Bass":
                    new_note.storedInstrument = instrument.Bass()
                elif instName is "Guitar":
                    new_note.storedInstrument = instrument.Guitar()
                elif instName is "Drums":
                    new_note.storedInstrument = instrument.ElectricOrgan() # THIS IS HACKISH!!!
                else:
                    new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            else:
                i += 1
    return output_notes


if __name__ == '__main__':
    wantedInstruments = ["Bass","Guitar","Drums"]
    database_npy = "midis_array_MultIntTestBGD_XL"
    matname = "multiInstAttBGD_XL"
    midisfolder = "midiFilesFor10701/*/*mid"
    perform_matrix_to_midi = True;


    ## Test midi_to_matrix and matrix_to_midi for single file
    band_name = 'Nirvana'
    song_name = 'Breed'
    filename_midi = 'midiFilesFor10701/' + band_name + '/' + song_name + '.mid'

    my_file_single_midi = Path(filename_midi)

    if my_file_single_midi.is_file():
        result = midi_to_matrix(filename_midi, 1500, wantedInstruments);
        ## matrix to midi doesn't work with these weird lists atm... needs to be fixed.
        # output_notes = matrix_to_midi(result)
        # midi_stream = stream.Stream(output_notes)
        # output_filename = song_name + 'bass.mid'
        # midi_stream.write('midi', fp=output_filename)
    else:
        print ("%s is not in directory" % filename_midi)
        

    plt.imshow(result[0][:][:]) # just to see it

	# remove singleton dimensions
    if len(result) == 1:        
        result = result[0]

	# Build database
	# Sources : https://chsasank.github.io/keras-tutorial.html
	#           https://stackoverflow.com/questions/28439701/how-to-save-and-load-numpy-array-data-properly


    my_file_database_original_npy = Path("./" + database_npy + '_original.npy')
    my_file_database_binary_npy = Path("./" + database_npy + '_binary.npy')
    my_file_database_simplified_npy = Path("./" + database_npy + '_simplified.npy')


    if my_file_database_original_npy.is_file(): 
        midis_array = np.load(my_file_database_original_npy)
        print("File already exists! Proceeding to MIDI instrument track decomposition")
        
    else:
        print (os.getcwd())
        root_dir = './'
        all_midi_paths = glob.glob(os.path.join(root_dir,midisfolder))
        print (all_midi_paths)
        matrix_of_all_midis = []
        total_count = len(all_midi_paths)

        # All midi have to be in same shape. 
        count = 0;
        for single_midi_path in all_midi_paths:
            count = count+1; # increment
            try:
                print ("Song " + str(count) + " of " + str(total_count))
                print (single_midi_path)
                matrix_of_single_midi = midi_to_matrix(single_midi_path, 1500, wantedInstruments)
                if len(matrix_of_single_midi) == 1: # reduce to single page if we can
                    matrix_of_single_midi = matrix_of_single_midi[0]
                if (matrix_of_single_midi is not None and sum(sum(sum(np.array(matrix_of_single_midi))))):
                    matrix_of_all_midis.append(matrix_of_single_midi)
                #print (matrix_of_single_midi.shape)
            except Exception as ex:
                continue;
            
            
        # remove any that are empty   # ADAM is this doing that...?
        midis_array_original = np.asarray(matrix_of_all_midis)
        # hack to make all binary and save both orig. and binary versions
        midis_array_binary = midis_array_original.copy(); 
        midis_array_binary[midis_array_binary != 0] = 1;
        # hack to yield only one note per timestep + binary
        midis_array_simplified = midis_array_binary.copy();
        
        if(np.size(midis_array_simplified.shape)==3): # single instrument
            lowest_notes = first_nonzero(midis_array_binary,axis=1,invalid_val=126)
            for i in range(len(midis_array_simplified)):
                for j in range(len(midis_array_simplified[0][0])):
                    replace_inds = np.arange(lowest_notes[i][j]+1,128).tolist()
                    for k in replace_inds: # this is stupid but I can't get it to work otherwise.
                        midis_array_simplified[i][k][j] = 0
        else:
            for h in range(np.size(midis_array_simplified,1)): # for each instrument
                lowest_notes = first_nonzero(midis_array_binary[:,h,:,:],axis=1,invalid_val=126)
                for i in range(np.size(lowest_notes,0)): # for each song
                    for j in range(np.size(lowest_notes,1)): # for each timestep
                        replace_inds = np.arange(lowest_notes[i,j]+1,128).tolist()
                        for k in replace_inds:
                            midis_array_simplified[i,h,k,j] = 0
        

        # make them smaller and save them
        midis_array_original = midis_array_original.astype(sp.float16)
        midis_array_binary = midis_array_binary.astype(sp.int8)
        midis_array_simplified = midis_array_simplified.astype(sp.int8)
        np.save(my_file_database_original_npy, midis_array_original)
        np.save(my_file_database_binary_npy, midis_array_binary)
        np.save(my_file_database_simplified_npy, midis_array_simplified)
        
        # for me so I can visualize easier...
        sio.savemat('./original'+matname+'.mat', mdict={'midis_array_original': midis_array_original})
        sio.savemat('./binary'+matname+'.mat',mdict={'midis_array_binary': midis_array_binary})
        sio.savemat('./simplified'+matname+'.mat',mdict={'midis_array_simplified': midis_array_simplified})

        print('Done with MIDI -> Matrix')
        
        
    if perform_matrix_to_midi:
        if len(wantedInstruments) > 1:
            for inst in wantedInstruments:
                for i in range(len(midis_array_original)): # for each song:
                    thePart = matrix_to_midi(midis_array_original[i,wantedInstruments.index(inst),:,:],inst)
                    midi_stream = stream.Stream(thePart)
                    songname = all_midi_paths[i].split("/",3)[2:4]
                    output_filename = wantedInstruments[0] + '_' + songname[0] + '_' + songname[1]
                    midi_stream.write('midi', fp=output_filename)
                    print("finished " + str(i) + " of " + str(len(midis_array_original)))   
        else:
            for i in range(len(midis_array_original)): # for each song:
                thePart = matrix_to_midi(midis_array_original[i,:,:],wantedInstruments[0])
                midi_stream = stream.Stream(thePart)
                songname = all_midi_paths[i].split("/",3)[2:4]
                output_filename = wantedInstruments[0] + '_' + songname[0] + '_' + songname[1]
                midi_stream.write('midi', fp=output_filename)
                print("finished " + str(i) + " of " + str(len(midis_array_original)))
