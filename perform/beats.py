"""Find a song's tempo and its first beat, for a choreography file.

    python perform/beats.py examples/my_song.mp3

Prints the `bpm:` and `first_beat:` lines to paste into the choreography. It assumes a
steady tempo, like most dance music.

How: the tempo is the one whose grid lines up best with every hit in the song, counted at
twice the tempo so that parts where the off-beats are loudest (claps, hi-hats) still
agree. Where the beat sits is then taken from the low kick drum alone, which is on it.
"""

from __future__ import annotations

import argparse

import librosa
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song")
    a = ap.parse_args()

    y, sr = librosa.load(a.song, sr=22050, mono=True)
    hop = 256
    hits = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    kicks = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=150, n_mels=32)
    t = librosa.frames_to_time(np.arange(len(hits)), sr=sr, hop_length=hop)
    rough = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=hits, sr=sr, hop_length=hop))[0])

    bpms = np.arange(0.92, 1.08, 0.0001) * rough
    fit = np.abs(np.exp(4j * np.pi * np.outer(bpms / 60.0, t)) @ hits)
    bpm = float(bpms[int(np.argmax(fit))])
    period = 60.0 / bpm
    phase = np.angle(np.exp(2j * np.pi * t / period) @ kicks)
    first = (phase / (2 * np.pi) * period) % period

    print(f"bpm: {bpm:.2f}")
    print(f"first_beat: {first:.3f}")


if __name__ == "__main__":
    main()
