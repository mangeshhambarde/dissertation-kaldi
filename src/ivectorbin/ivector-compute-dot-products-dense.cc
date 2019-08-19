// ivectorbin/ivector-compute-dot-products-dense.cc

// Copyright 2016-2018  David Snyder
//           2017-2018  Matthew Maciejewski

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/stl-utils.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
      "Perform cosine scoring for speaker diarization.  The input reco2utt\n"
      "should be of the form <recording-id> <seg1> <seg2> ... <segN> and\n"
      "there should be one iVector for each segment.  Cosine scoring is\n"
      "performed between all pairs of iVectors in a recording and outputs\n"
      "an archive of score matrices, one for each recording-id.  The rows\n"
      "and columns of the the matrix correspond the sorted order of the\n"
      "segments.\n"
      "Usage: ivector-compute-dot-products-dense.cc [options] <reco2utt>"
      " <ivectors-rspecifier> <scores-wspecifier>\n"
      "e.g.: \n"
      "  ivector-compute-dot-products-dense reco2utt scp:ivectors.scp"
      " ark:scores.ark ark,t:ivectors.1.ark\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string reco2utt_rspecifier = po.GetArg(1),
      ivector_rspecifier = po.GetArg(2),
      scores_wspecifier = po.GetArg(3);

    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    BaseFloatMatrixWriter scores_writer(scores_wspecifier);
    int32 num_reco_err = 0,
          num_reco_done = 0;
    for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
      std::string reco = reco2utt_reader.Key();

      std::vector<std::string> uttlist = reco2utt_reader.Value();
      std::vector<Vector<BaseFloat> > ivectors;

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];

        if (!ivector_reader.HasKey(utt)) {
          KALDI_ERR << "No iVector present in input for utterance " << utt;
        }

        Vector<BaseFloat> ivector = ivector_reader.Value(utt);
        ivectors.push_back(ivector);
      }
      if (ivectors.size() == 0) {
        KALDI_WARN << "Not producing output for recording " << reco
                   << " since no segments had iVectors";
        num_reco_err++;
      } else {
        Matrix<BaseFloat> scores(ivectors.size(), ivectors.size());
        for (int32 i = 0; i < scores.NumRows(); i++) {
          for (int32 j = 0; j < scores.NumRows(); j++) {
            scores(i, j) = VecVec(ivectors[i], ivectors[j]);
          }
        }
        scores_writer.Write(reco, scores);
        num_reco_done++;
      }
    }
    KALDI_LOG << "Processed " << num_reco_done << " recordings, "
              << num_reco_err << " had errors.";
    return (num_reco_done != 0 ? 0 : 1 );
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
