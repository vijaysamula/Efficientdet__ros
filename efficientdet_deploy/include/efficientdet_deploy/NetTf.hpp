#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/framework/tensor_slice.h"

#include <efficientdet_deploy/utils.h>

using namespace std;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::tstring;
using tensorflow::SavedModelBundle;
using tensorflow::SessionOptions;
using tensorflow::RunOptions;
using tensorflow::Scope;
using tensorflow::ClientSession;

struct Prediction{
	std::unique_ptr<std::vector<std::vector<float>>> boxes;
	std::unique_ptr<std::vector<float>> scores;
	std::unique_ptr<std::vector<int>> labels;
};

namespace efficientdet {

class ModelLoader{
	private:
		SavedModelBundle bundle;
		SessionOptions session_options;
		RunOptions run_options;
		
		void make_prediction(Tensor &image_output, std::vector<Tensor> &predictions,double thresholdIOU,double thresholdScore);
	public:
		//ModelLoader(const string& model_path,const string& model_path_txt,map<int, string>& labelsMap);
		ModelLoader(const string& model_path,const float& mem_percentage, bool saved_model);

		Status loadGraph(const std::string &graph_file_name,std::unique_ptr<tensorflow::Session> *session); 
		void predict(const cv::Mat& image, std::vector<Tensor> &predictions,double thresholdIOU,double thresholdScore,bool gray =false);
        Status ReadImage(const cv::Mat& image, Tensor &outTensor);
        //Status readLabelsMapFile(const string &fileName, map<int, string> &labelsMap);
		
		std::vector<std::string> classes;
};
}