#!/usr/bin/env python3
"""TODO."""
from __future__ import print_function
from __future__ import unicode_literals
import argparse
from collections import defaultdict, namedtuple
import glob
import itertools
import os
import sys


Segment = namedtuple('Segment', ['id', 'onset', 'offset', 'label'])
Turn = namedtuple(
    'Turn', ['turn_id', 'type', 'fid', 'channel', 'onset', 'dur', 'ortho', 'speaker_type',
             'speaker_id', 'score', 'slt'])

def load_rttm(fn):
    """Load turns from RTTM file."""
    with open(fn, 'rb') as f:
        turns = []
        for n, line in enumerate(f):
            fields = line.decode('utf-8').strip().split()
            rec_id = fields[1]
            turn_id = '{}_{}'.format(rec_id, str(n).zfill(4))
            fields = [turn_id] + fields
            turns.append(Turn(*fields))
    return turns


def write_rttm(fn, turns):
    """Write turns to RTTM file."""
    with open(fn, 'wb') as f:
        turns = sorted(
            turns, key=lambda x: (x.fid, float(x.onset), float(x.dur)))
        for turn in turns:
            line = ' '.join(turn[1:])
            f.write(line.encode('utf-8'))
            f.write(b'\n')

def fix_speakers(rec_to_turns):
    """ Make speaker ids uniform. """
    speaker_count = 0
    speaker_map = {}
    for rec in rec_to_turns:
        for turn in rec_to_turns[rec]:
            sid = turn.speaker_id
            if sid not in speaker_map:
                speaker_count += 1
                speaker_map[sid] = "speaker_" + str(speaker_count).zfill(4)
    # replace with new ids.
    new_rec_to_turns = {}
    for rec in rec_to_turns:
        new_turns = []
        for turn in rec_to_turns[rec]:
            new_turn = turn._replace(speaker_id=speaker_map[turn.speaker_id])
            new_turns.append(new_turn)
        new_rec_to_turns[rec] = new_turns
    return new_rec_to_turns

def prefix_speaker_ids(rec_to_turns):
    new_rec_to_turns = {}
    for rec in rec_to_turns:
        new_turns = []
        for turn in rec_to_turns[rec]:
            new_turn = turn._replace(turn_id=turn.speaker_id + "-" + turn.turn_id)
            new_turns.append(new_turn)
        new_rec_to_turns[rec] = new_turns
    return new_rec_to_turns

def write_wav_scpf(fn, turns, audio_dir, audio_ext='.flac'):
    """Write script file containing WAV data for speech segments.

    Parameters
    ----------
    fn : str
        Path to output script file.

    turns : list of str
        List of unique identifiers.

    audio_dir : str
        Path to directory containing audio files.

    audio_ext : str, optional
        Audio file extension.
        (Default: '.flac')
    """
    with open(fn, 'wb') as f:
        for turn in sorted(turns):
            if audio_ext == '.flac':
                wav_str = ('{} sox -t flac {}/{}.flac -t wav -r 16k '
                           '-b 16 --channels 1 - |\n'.format(turn, audio_dir, turn))
            elif audio_ext == '.wav':
                wav_str = ('{} sox -t wav {}/{}.wav -t wav -r 16k '
                           '-b 16 --channels 1 - |\n'.format(turn, audio_dir, turn))
            f.write(wav_str.encode('utf-8'))


def write_utt2spk(fn, rec_to_turns):
    """Write ``utt2spk`` file."""
    file_lines = []
    for rec in rec_to_turns:
        turns = rec_to_turns[rec]
        for turn in turns:
            line = '{} {}\n'.format(turn.turn_id, turn.speaker_id)
            file_lines.append(line)
    with open(fn, 'wb') as f:
        for line in sorted(file_lines):
            f.write(line.encode('utf-8'))

def write_segments_file(fn, rec_to_turns):
    """Write ``segments`` file."""
    all_lines = []
    for rec in sorted(rec_to_turns):
        turns = sorted(
            rec_to_turns[rec], key=lambda x: x.turn_id)
        for turn in turns:
            line = '{} {} {} {}\n'.format(
                turn.turn_id, rec, turn.onset, float(turn.onset) + float(turn.dur))
            all_lines.append(line)
    with open(fn, 'wb') as f:
        for line in sorted(all_lines):
            f.write(line.encode('utf-8'))


def get_rec(fn):
    """Get recording corresponding to filename."""
    return os.path.splitext(os.path.basename(fn))[0]


def write_rec2num_spk(fn, rec_to_turns):
    """Write ``rec2num_spk``."""
    rec_to_speakers = defaultdict(set)
    for rec, turns in rec_to_turns.items():
        for turn in turns:
            rec_to_speakers[rec].add(turn.speaker_id)
    with open(fn, 'wb') as f:
        for rec in sorted(rec_to_speakers):
            n_speakers = len(rec_to_speakers[rec])
            line = '{} {}\n'.format(rec, n_speakers)
            f.write(line.encode('utf-8'))


def prepare_data_dir(data_dir, sad_dir, audio_dir, rttm_dir=None,
                     audio_ext='.flac', sad_ext='.lab', rttm_ext='.rttm'):
    """Prepare data directory.

    This function will create the following files in ``data_dir``:

    - wav.scp  --  script mapping audio to WAV data suitable for feature
      extraction
    - utt2spk  --  mapping from audio files to segment ids
    - segments  --  listing of **ALL** speech segments in source recordings
      according to segmentations from label files under ``sad_dir``
    - rttm  --  combined RTTM file created from contents of RTTM files under
      ``rttm_dir``; not written if ``rttm_dir`` is None
    - reco2num_spk  --  mapping from audio files to number of reference
      speakers present; not written if ``rttm_dir`` is None

    Parameters
    ----------
    data_dir : str
        Path to output directory.

    sad_dir : str
        Path to directory containing SAD label files. Assumes all files have
        extension ``.lab``.

    audio_dir : str
        Path to directory containing audio files.

    rttm_dir : str, optional
        Path to directory containing RTTM files.
        (Default: None)

    audio_ext : str, optional
        Audio file extension. Must be one of {'.wav', '.flac'}.
        (Default: '.flac')

    sad_ext : str, optional
        SAD file extension.
        (Default: '.lab')

    rttm_ext : str, optional
        RTTM file extension.
        (Default: '.rttm')
    """
    # Load turns from rttm files.
    rec_to_turns = {}
    for filename in glob.glob(os.path.join(rttm_dir, '*' + rttm_ext)):
        turns = load_rttm(filename)
        rec_to_turns[get_rec(filename)] = turns

    rec_to_turns = fix_speakers(rec_to_turns)
    rec_to_turns = prefix_speaker_ids(rec_to_turns)

    # Write wav.scp.
    write_wav_scpf(
        os.path.join(data_dir, 'wav.scp'),
        rec_to_turns.keys(), audio_dir, audio_ext)

    # Write the combined RTTM and reference num speakers files.
    combined_turns = list(itertools.chain.from_iterable(
        rec_to_turns.values()))
    write_rttm(
        os.path.join(data_dir, 'rttm'), combined_turns)
    write_rec2num_spk(
        os.path.join(data_dir, 'reco2num_spk'), rec_to_turns)

    # Write utt2spk and segments.
    write_utt2spk(
        os.path.join(data_dir, 'utt2spk'), rec_to_turns)
    write_segments_file(
        os.path.join(data_dir, 'segments'), rec_to_turns)


def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description='Prepare data directory for KALDI experiments.',
        add_help=True)
    parser.add_argument(
        'data_dir', nargs=None, help='output data directory')
    parser.add_argument(
        'audio_dir', nargs=None, help='source audio directory')
    parser.add_argument(
        'sad_dir', nargs=None, help='source SAD directory')
    parser.add_argument(
        '--rttm_dir', nargs=None, default=None, metavar='STR',
        help='source RTTM directory')
    parser.add_argument(
        '--audio_ext', nargs=None, default='.flac', metavar='STR',
        choices=['.flac', '.wav'],
        help='audio file extension (default: %(default)s)')
    parser.add_argument(
        '--sad_ext', nargs=None, default='.lab', metavar='STR',
        help='SAD file extension (default: %(default)s)')
    parser.add_argument(
        '--rttm_ext', nargs=None, default='.rttm', metavar='STR',
        help='RTTM file extension (default: %(default)s)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    prepare_data_dir(
        args.data_dir, args.sad_dir, args.audio_dir, args.rttm_dir,
        args.audio_ext, args.sad_ext, args.rttm_ext)



if __name__ == '__main__':
    main()
