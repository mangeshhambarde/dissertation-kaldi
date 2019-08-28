#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.
# submitjob -p MINI -q LONGGPU -m 50000 -o -l longjob=1 -l hostname=node22 -eo test.log ./run.sh

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
VOXCELEB_DIR=/share/mini5/data/audvis/dia/voxceleb/voxceleb1

nnet_dir=exp/xvector_nnet_1a

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
  # segmentation for dihard (rttm files) but not for voxceleb.
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    --vad-config conf/vad-all-speech.conf \
    ${DEV_DATA_DIR} exp/make_vad $vaddir
  utils/fix_data_dir.sh ${DEV_DATA_DIR}

  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    --vad-config conf/vad.conf \
    ${VOXCELEB_DATA_DIR} exp/make_vad $vaddir
  utils/fix_data_dir.sh ${VOXCELEB_DATA_DIR}

  # Combine both.
  utils/combine_data.sh data/train_with_sil $DEV_DATA_DIR $VOXCELEB_DATA_DIR
fi
echo "Stage 1: Extract MFCC and compute VAD done."

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    data/train_with_sil data/train exp/train
  cp data/train_with_sil/segments data/train
  cp data/train_with_sil/vad.scp data/train
  utils/fix_data_dir.sh data/train
fi

if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=400
  mv data/train/utt2num_frames data/train/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train/utt2num_frames.bak > data/train/utt2num_frames
  utils/filter_scp.pl data/train/utt2num_frames data/train/utt2spk > data/train/utt2spk.new
  mv data/train/utt2spk.new data/train/utt2spk
  utils/fix_data_dir.sh data/train

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=4
  awk '{print $1, NF-1}' data/train/spk2utt > data/train/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train/spk2num | utils/filter_scp.pl - data/train/spk2utt > data/train/spk2utt.new
  mv data/train/spk2utt.new data/train/spk2utt
  utils/spk2utt_to_utt2spk.pl data/train/spk2utt > data/train/utt2spk

  utils/filter_scp.pl data/train/utt2spk data/train/utt2num_frames > data/train/utt2num_frames.new
  mv data/train/utt2num_frames.new data/train/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/train
fi

echo "Stage 2-5: Remove too short utterances, speakers with few utterances done."

# Stages 6 through 8 are handled in run_xvector.sh
local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 \
  --data data/train --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs

echo "Stage 6-8: Train nnet done."

if [ $stage -le 9 ]; then
  # Extract x-vectors for centering, LDA, and PLDA training.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
    $nnet_dir data/train \
    $nnet_dir/xvectors_train
fi

echo "Stage 9: Extract x-vectors done."

if [ $stage -le 10 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $nnet_dir/xvectors_train/log/compute_mean.log \
    ivector-mean scp:$nnet_dir/xvectors_train/xvector.scp \
    $nnet_dir/xvectors_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd $nnet_dir/xvectors_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train/xvector.scp ark:- |" \
    ark:data/train/utt2spk $nnet_dir/xvectors_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnet_dir/xvectors_train/log/plda.log \
    ivector-compute-plda ark:data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnet_dir/xvectors_train/plda || exit 1;
fi

echo "Stage 10: Train PLDA done."
