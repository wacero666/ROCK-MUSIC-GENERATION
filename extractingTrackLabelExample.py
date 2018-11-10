#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:13:56 2018

Example of extracting relevant track label information

@author: adamsmoulder
"""

from music21 import midi


# Track labels should be stored in the midiProgram data of parsed streams, 
# though for some odd reason music21 misreads this a lot... instead, we can
# work around it by looking at program changes in the MIDI file events
#
# Outputs are trackChannels, trackInstrumentIds, and emptyTrackIndices

# parse the given MIDI file
midiFilePath = 'midiFilesFor10701/FooFighters/MyHero.mid'
mf = midi.MidiFile()  
mf.open(midiFilePath)
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
        trackInstrumentIds.append(t.getProgramChanges()[0])
    else:
        trackChannels.append(-1)
        trackInstrumentIds.append(-1)
        emptyTrackIndices.append(count)
    count += 1;



# sidenote: normal data for use with music21 can be read in using commands like
#
# midiStream = converter.parse(midiFilePath)
# 
# in this, midiStream is a list of streams containing each of the tracks 
# that have notes in order (empty tracks are elminated). Streams actually have
# a lot of stuff to work with in music21 if we use it; worst case, this script
# at least gets the instrument ids!

    