#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#           2018  David Snyder
#
# Usage: make_voxceleb1.pl /export/voxceleb1 data/
#
use File::Find qw(finddepth);

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-voxceleb1> <path-to-data-dir>\n";
  print STDERR "e.g. $0 /export/voxceleb1 data/\n";
  exit(1);
}

($data_base, $out_dir) = @ARGV;
my $data_dir = "$out_dir/voxceleb1_full";

if (system("mkdir -p $data_dir") != 0) {
  die "Error making directory $data_dir";
}

opendir my $dh, "$data_base/voxceleb1_wav" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$data_base/voxceleb1_wav/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

if (! -e "$data_base/vox1_meta.csv") {
  system("wget -O $data_base/vox1_meta.csv http://www.openslr.org/resources/49/vox1_meta.csv");
}

open(META_IN, "<", "$data_base/vox1_meta.csv") or die "Could not open the meta data file $data_base/vox1_meta.csv";
open(SPKR_OUT, ">", "$data_dir/utt2spk") or die "Could not open the output file $data_dir/utt2spk";
open(WAV_OUT, ">", "$data_dir/wav.scp") or die "Could not open the output file $data_dir/wav.scp";

my %id2spkr = ();
while (<META_IN>) {
  chomp;
  my ($vox_id, $spkr_id, $gender, $nation, $set) = split;
  $id2spkr{$vox_id} = $spkr_id;
}

foreach (@spkr_dirs) {
  my $spkr_id = $_;
  my $new_spkr_id = $spkr_id;
  # If we're using a newer version of VoxCeleb1, we need to "deanonymize"
  # the speaker labels.
  if (exists $id2spkr{$spkr_id}) {
    $new_spkr_id = $id2spkr{$spkr_id};
  }
  my @files;
  finddepth(sub {
        return if($_ eq '.' || $_ eq '..');
        if ($_ =~ /\.wav$/) {
          push @files, $File::Find::name;
        }
   }, "$data_base/voxceleb1_wav/$spkr_id/");
  foreach (@files) {
    my $filename = substr($_, -9, 9);
    my $rec_id = substr($_, -21, 11);
    my $segment = substr($_, -9, 5);
    my $wav = "$_";
    my $utt_id = "$new_spkr_id-$rec_id-$segment";
    print WAV_OUT "$utt_id", " $wav", "\n";
    print SPKR_OUT "$utt_id", " $new_spkr_id", "\n";
  }
}

close(SPKR_OUT) or die;
close(WAV_OUT) or die;
close(META_IN) or die;

if (system(
  "utils/utt2spk_to_spk2utt.pl $data_dir/utt2spk >$data_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $data_dir";
}
system("env LC_COLLATE=C utils/fix_data_dir.sh $data_dir");
if (system("env LC_COLLATE=C utils/validate_data_dir.sh --no-text --no-feats $data_dir") != 0) {
  die "Error validating directory $data_dir";
}
