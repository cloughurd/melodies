import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
from collections import Counter

def plot_piano_roll(pm, title="Piano Roll", max_time=30):
    plt.figure(figsize=(12, 6))
    for instrument in pm.instruments:
        pitches = [note.pitch for note in instrument.notes if note.start < max_time]
        starts = [note.start for note in instrument.notes if note.start < max_time]
        durations = [note.end - note.start for note in instrument.notes if note.start < max_time]
        plt.barh(pitches, durations, left=starts, height=0.6, alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("MIDI Pitch")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_note_density(pm, title="Note Density", interval=1.0, max_time=30):
    note_starts = [note.start for inst in pm.instruments for note in inst.notes if note.start < max_time]
    bins = np.arange(0, max_time + interval, interval)
    counts, _ = np.histogram(note_starts, bins)
    plt.figure(figsize=(10, 4))
    plt.plot(bins[:-1], counts, drawstyle='steps-post')
    plt.xlabel("Time (s)")
    plt.ylabel("Notes Played")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_pitch_histogram(pm, title="Pitch Histogram"):
    pitches = [note.pitch for inst in pm.instruments for note in inst.notes]
    pitch_counts = Counter(pitches)
    plt.figure(figsize=(10, 4))
    plt.bar(pitch_counts.keys(), pitch_counts.values())
    plt.xlabel("MIDI Pitch")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.show()


def visualize_song(song_name, cluster_df, bitmidi_map, maestro_map):
    match = cluster_df[cluster_df['song'] == song_name]
    source = match.iloc[0]['source']
    file_name = match.iloc[0]['filename']
    try:
        if source == 'bitmidi':
            pm = bitmidi_map[file_name]
        elif source == 'maestro':
            pm = maestro_map[file_name]
        plot_piano_roll(pm, title=f"Piano Roll: {song_name}")
        plot_note_density(pm, title=f"Note Density: {song_name}")
        plot_pitch_histogram(pm, title=f"Pitch Histogram: {song_name}")
    except Exception as e:
        print(f"Error visualizing '{song_name}': {e}")

