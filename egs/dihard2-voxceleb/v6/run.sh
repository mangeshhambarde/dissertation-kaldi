#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.
#
# Seems to have worked with nj=1, nt=4, np=10:
# submitjob -p MINI -q LONGGPU -m 200000 -o -l longjob=1 -l hostname=node21 -eo test.log ./run.sh

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc

export PATH="/share/spandh.ami1/sw/std/python/anaconda3-5.1.0/v5.1.0/bin:$PATH" # for virtualenv

# Path to root of DIHARD 2018 releases
DIHARD_DEV_DIR=/share/mini5/data/audvis/dia/dihard-2018-dev-for-use-with-2019-baseline
DIHARD_EVAL_DIR=/share/mini5/data/audvis/dia/dihard-2018-eval-for-use-with-2019-baseline

# Path to Voxceleb I.
VOXCELEB_FULL_DIR=/share/mini5/data/audvis/dia/voxceleb/voxceleb1

# Path to trained models.
MODEL_DIR=exp/models

stage=0

if [ $stage -le 0 ]; then
  # Delete old data directories.
  rm -fr data/dihard*
  rm -fr data/voxceleb*
  rm -fr data/train*

  # Prepare data directory for Dihard.
  echo "Preparing data directory for DEV set..."
  local/make_data_dir.py \
     --audio_ext '.flac' \
     --rttm_dir $DIHARD_DEV_DIR/data/single_channel/rttm \
     data/dihard_24 \
     $DIHARD_DEV_DIR/data/single_channel/flac \
     $DIHARD_DEV_DIR/data/single_channel/sad
  utils/fix_data_dir.sh data/dihard_24

  # Prepare data directory for Voxceleb.
  echo "Preparing data directory for Voxceleb..."
  local/make_voxceleb1.pl \
     $VOXCELEB_FULL_DIR data/voxceleb_24
  utils/fix_data_dir.sh data/voxceleb_24

  # Copy to make data directories for 30 dim mfcc.
  utils/copy_data_dir.sh data/dihard_24 data/dihard_30
  utils/copy_data_dir.sh data/voxceleb_24 data/voxceleb_30
fi

echo "Stage 0: Prepare data directory done."

if [ $stage -le 1 ]; then
  # Make 24 dim MFCCs.
  for dir in dihard_24 voxceleb_24; do
    steps/make_mfcc.sh --write-utt2num-frames true \
      --mfcc-config conf/mfcc-24.conf --nj 40 --cmd "$train_cmd" \
      data/$dir exp/make_mfcc_24 $mfccdir
    utils/fix_data_dir.sh data/$dir
  done

  # Make 30 dim MFCCs.
  for dir in dihard_30 voxceleb_30; do
    steps/make_mfcc.sh --write-utt2num-frames true \
      --mfcc-config conf/mfcc-30.conf --nj 40 --cmd "$train_cmd" \
      data/$dir exp/make_mfcc_30 $mfccdir
    utils/fix_data_dir.sh data/$dir
  done

  # Compute VAD separately for both datasets, because we already have gold
  # speech segmentation for dihard (label files) but not for voxceleb.
  # Compute only for 24 dim, then copy to 30 dim.
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    --vad-config conf/vad-all-speech.conf \
    data/dihard_24 exp/make_vad $mfccdir
  utils/fix_data_dir.sh data/dihard_24
  cp data/dihard_24/vad.scp data/dihard_30

  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    --vad-config conf/vad.conf \
    data/voxceleb_24 exp/make_vad $mfccdir
  utils/fix_data_dir.sh data/voxceleb_24
  cp data/voxceleb_24/vad.scp data/voxceleb_30

  # Combine both.
  utils/combine_data.sh data/train_24 data/dihard_24 data/voxceleb_24
  utils/combine_data.sh data/train_30 data/dihard_30 data/voxceleb_30
fi
echo "Stage 1: Extract MFCC and compute VAD done."

if [ $stage -le 2 ]; then
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 40G" --nj 80 \
    $MODEL_DIR data/train_24 \
    exp/ivectors_train
fi
echo "Stage 2: Extracting i-vectors done."

if [ $stage -le 3 ]; then
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
    $MODEL_DIR data/train_30 \
    exp/xvectors_train
fi
echo "Stage 3: Extract x-vectors done."

if [ $stage -le 4 ]; then
  mkdir -p exp/cvectors_train
  # Concatenate i-vectors and x-vectors to get c-vectors.
  $train_cmd exp/cvectors_train/log/paste_feats.log \
    append-vectors scp:exp/ivectors_train/ivector.scp \
      scp:exp/xvectors_train/xvector.scp \
      ark,scp:exp/cvectors_train/cvector.ark,exp/cvectors_train/cvector.scp || exit 1;
fi
echo "Stage 4: Concatenation done."

if [ $stage -le 5 ]; then
  # Compute the mean vector for centering the evaluation c-vectors.
  $train_cmd exp/cvectors_train/log/compute_mean.log \
    ivector-mean scp:exp/cvectors_train/cvector.scp \
    exp/cvectors_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd exp/cvectors_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/cvectors_train/cvector.scp ark:- |" \
    ark:data/train_24/utt2spk exp/cvectors_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd exp/cvectors_train/log/plda.log \
    ivector-compute-plda ark:data/train_24/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/cvectors_train/cvector.scp ark:- | transform-vec exp/cvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/cvectors_train/plda || exit 1;
fi
