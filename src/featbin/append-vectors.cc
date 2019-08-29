// featbin/append-vector-to-feats.cc

// Copyright 2012 Korbinian Riedhammer
//           2013 Brno University of Technology (Author: Karel Vesely)
//           2013-2014 Johns Hopkins University (Author: Daniel Povey)

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;

    const char *usage =
        "Concatenate two vectors\n"
        "\n"
        "Usage: append-vectors <vec-rspecifier1> <vec-rspecifier2> <out-wspecifier>\n"
        "See also: paste-feats, concat-feats\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    string vec1_rspecifier = po.GetArg(1);
    string vec2_rspecifier = po.GetArg(2);
    string out_wspecifier = po.GetArg(3);

    SequentialBaseFloatVectorReader vec1_reader(vec1_rspecifier);
    SequentialBaseFloatVectorReader vec2_reader(vec2_rspecifier);

    BaseFloatVectorWriter out_writer(out_wspecifier);

    int32 num_done = 0;
    // Main loop
    for (; !vec1_reader.Done() && !vec2_reader.Done(); vec1_reader.Next(), vec2_reader.Next()) {
      string utt1 = vec1_reader.Key();
      string utt2 = vec2_reader.Key();

      if (utt1 != utt2) {
        KALDI_ERR << "Mismatched utterances " << utt1 << " and " << utt2;
        exit(1);
      }
          
      const Vector<BaseFloat> &vec1 = vec1_reader.Value();
      const Vector<BaseFloat> &vec2 = vec2_reader.Value();

      Vector<BaseFloat> out_vec;
      out_vec.Resize(vec1.Dim() + vec2.Dim());
      int32 i = 0;
      for(int32 j = 0; j < vec1.Dim(); j++) {
          out_vec(i++) = vec1(j);
      }
      for(int32 j = 0; j < vec2.Dim(); j++) {
          out_vec(i++) = vec2(j);
      }
      out_writer.Write(utt1, out_vec);
      num_done++;
    }
    KALDI_LOG << "Done " << num_done << " utts.";

    if (num_done == 0 || !vec1_reader.Done() || !vec2_reader.Done()) {
        return -1;
    } else {
        return 0;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
