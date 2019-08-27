#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#
# Usage: make_voxceleb2.pl /export/voxceleb2 data/train
#
# Note: This script requires ffmpeg to be installed and its location included in $PATH.

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-voxceleb2> <path-to-data-dir>\n";
  print STDERR "e.g. $0 /export/voxceleb2 data\n";
  exit(1);
}

# Check that ffmpeg is installed.
if (`which ffmpeg` eq "") {
  die "Error: this script requires that ffmpeg is installed.";
}

($data_base, $out_dir) = @ARGV;
my $data_dir = "$out_dir/voxceleb2_full";

opendir my $dh, "$data_base/aac" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$data_base/aac/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

if (system("mkdir -p $data_dir") != 0) {
  die "Error making directory $data_dir";
}

open(SPKR, ">", "$data_dir/utt2spk") or die "Could not open the output file $data_dir/utt2spk";
open(WAV, ">", "$data_dir/wav.scp") or die "Could not open the output file $data_dir/wav.scp";

foreach (@spkr_dirs) {
  my $spkr_id = $_;

  opendir my $dh, "$data_base/aac/$spkr_id/" or die "Cannot open directory: $!";
  my @rec_dirs = grep {-d "$data_base/aac/$spkr_id/$_" && ! /^\.{1,2}$/} readdir($dh);
  closedir $dh;

  foreach (@rec_dirs) {
    my $rec_id = $_;

    opendir my $dh, "$data_base/aac/$spkr_id/$rec_id/" or die "Cannot open directory: $!";
    my @files = map{s/\.[^.]+$//;$_}grep {/\.m4a$/} readdir($dh);
    closedir $dh;

    foreach (@files) {
      my $name = $_;
      my $wav = "ffmpeg -v 8 -i $data_base/aac/$spkr_id/$rec_id/$name.m4a -f wav -acodec pcm_s16le -|";
      my $utt_id = "$spkr_id-$rec_id-$name";
      print WAV "$utt_id", " $wav", "\n";
      print SPKR "$utt_id", " $spkr_id", "\n";
    }
  }
}
close(SPKR) or die;
close(WAV) or die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $data_dir/utt2spk >$data_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $data_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $data_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $data_dir") != 0) {
  die "Error validating directory $data_dir";
}
