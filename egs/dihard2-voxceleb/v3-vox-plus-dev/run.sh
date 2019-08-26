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
# Run as: submitjob -p MINI -q LONGGPU -o -l longjob=1 -eo test.log ./run.sh

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

export PATH="/share/spandh.ami1/sw/std/python/anaconda3-5.1.0/v5.1.0/bin:$PATH" # for virtualenv

# Path to root of DIHARD 2018 releases
DIHARD_DEV_DIR=/share/mini5/data/audvis/dia/dihard-2018-dev-for-use-with-2019-baseline
DIHARD_EVAL_DIR=/share/mini5/data/audvis/dia/dihard-2018-eval-for-use-with-2019-baseline

# Path to Voxceleb I.
VOXCELEB_DIR=/share/spandh.ami1/embed-stud-proj/2018/mangesh/ws/dissertation/corpora/voxceleb/voxceleb1

dihard_dev=dihard_dev_2018
voxceleb=voxceleb1_full

stage=0

if [ $stage -le 0 ]; then
  # Prepare data directory for DEV set.
  echo "Preparing data directory for DEV set..."
  DEV_DATA_DIR=data/${dihard_dev}
  rm -fr $DEV_DATA_DIR
  local/make_data_dir.py \
     --audio_ext '.flac' \
     --rttm_dir $DIHARD_DEV_DIR/data/single_channel/rttm \
     $DEV_DATA_DIR \
     $DIHARD_DEV_DIR/data/single_channel/flac \
     $DIHARD_DEV_DIR/data/single_channel/sad
  utils/fix_data_dir.sh $DEV_DATA_DIR

  # Prepare data directory for Voxceleb.
  echo "Preparing data directory for Voxceleb..."
  VOXCELEB_DATA_DIR=data/${voxceleb}
  rm -fr $VOXCELEB_DATA_DIR
  local/make_voxceleb1.pl \
     $VOXCELEB_DIR data/
  utils/fix_data_dir.sh $VOXCELEB_DATA_DIR
fi

echo "Stage 0: Prepare data directory done."

if [ $stage -le 1 ]; then
  # Make MFCCs for both data dirs.
  steps/make_mfcc.sh --write-utt2num-frames true \
    --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    ${DEV_DATA_DIR} exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh ${DEV_DATA_DIR}

  steps/make_mfcc.sh --write-utt2num-frames true \
    --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    ${VOXCELEB_DATA_DIR} exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh ${VOXCELEB_DATA_DIR}

  # Compute VAD separately for both, because we already have gold speech
  # segmentation for dihard (label files) but not for voxceleb.
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    --vad-config conf/vad-all-speech.conf \
    ${DEV_DATA_DIR} exp/make_vad $vaddir
  utils/fix_data_dir.sh ${DEV_DATA_DIR}

  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    --vad-config conf/vad.conf \
    ${VOXCELEB_DATA_DIR} exp/make_vad $vaddir
  utils/fix_data_dir.sh ${VOXCELEB_DATA_DIR}

  # Combine both.
  utils/combine_data.sh data/train $DEV_DATA_DIR $VOXCELEB_DATA_DIR
fi

echo "Stage 1: Extract MFCC and compute VAD done."

if [ $stage -le 2 ]; then
  # Train the UBM.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 4G" \
    --nj 40 --num-threads 8 \
    data/train 2048 \
    exp/diag_ubm

  sid/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
    --nj 40 --remove-low-count-gaussians false \
    data/train \
    exp/diag_ubm exp/full_ubm
fi

echo "Stage 2: Train UBM done."

if [ $stage -le 3 ]; then
  # Train the i-vector extractor.
  sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 40G" \
    --ivector-dim 400 --num-iters 5 \
    --nj 1 --num-threads 4 --num-processes 8 \
    exp/full_ubm/final.ubm data/train \
    exp/extractor
fi

echo "Stage 3: Train i-vector extractor done."

if [ $stage -le 4 ]; then
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 40G" --nj 80 \
    exp/extractor data/train \
    exp/ivectors_train
fi

echo "Stage 4: Extracting i-vectors done."

if [ $stage -le 5 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_train/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_train/ivector.scp \
    exp/ivectors_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd exp/ivectors_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train/ivector.scp ark:- |" \
    ark:data/train/utt2spk exp/ivectors_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd exp/ivectors_train/log/plda.log \
    ivector-compute-plda ark:data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train/ivector.scp ark:- | transform-vec exp/ivectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/ivectors_train/plda || exit 1;
fi
