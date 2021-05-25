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

#include <opencv2/core/mat.hpp>


#include <efficientdet_deploy/NetTf.hpp>
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
using namespace cv;



namespace efficientdet {
ModelLoader::ModelLoader(const string& model_path ,const float& mem_percentage, bool saved_model){	
	if (saved_model) {
		session_options.config.mutable_gpu_options()->set_allow_growth(true);
		session_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(mem_percentage);	
		auto status = tensorflow::LoadSavedModel(session_options, run_options, model_path, {"serve"},
				&bundle);

		if (status.ok()){
			printf("Model loaded successfully...\n");
		}
		else {
			printf("Error in loading model\n");
		}
	 }
	else {
		// Load and initialize the model from .pb file
		std::unique_ptr<tensorflow::Session> session;
		LOG(INFO) << "graphPath:" << model_path;
		Status loadGraphStatus = loadGraph(model_path, &session);
		if (!loadGraphStatus.ok()) {
			LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
		} else
			LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;
	}


	
	
}

/** Read a model graph definition (xxx.pb) from disk, and creates a session object you can use to run it.
 */
Status ModelLoader::loadGraph(const std::string &graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}



void ModelLoader::predict(const cv::Mat& image, std::vector<Tensor> &predictions,double thresholdIOU=0.8,double thresholdScore=0.5){
	// Get dimensions
	unsigned int cv_img_h = image.rows;
	unsigned int cv_img_w = image.cols;
	unsigned int cv_img_d = image.channels();

	Tensor image_output(tensorflow::DT_FLOAT, {1, cv_img_h, cv_img_w, cv_img_d});
	auto read_status = ReadImage(image, image_output);
	make_prediction(image_output, predictions,thresholdIOU,thresholdScore);
}

void ModelLoader::make_prediction(Tensor &image_output, std::vector<Tensor> &predictions,double thresholdIOU,double thresholdScore){
	const string input_node = "image_arrays:0";
	
	
	std::vector<std::pair<string, Tensor>> inputs_data  = {{input_node, image_output}};
	std::vector<string> output_nodes = {{"detections:0"}};

	this->bundle.GetSession()->Run(inputs_data, output_nodes, {}, &predictions);
}

Status ModelLoader::ReadImage(const cv::Mat& image, Tensor &outTensor){

	auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

	// tf pointer for init of fake cv mat
	float* outTensor_pointer = outTensor.flat<float>().data();

	// fake cv mat (avoid copy)
	cv::Mat fakemat_cv(image.rows, image.cols, CV_32FC3, outTensor_pointer);
	image.convertTo(fakemat_cv, CV_32FC3);

    auto input_tensor = Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
    vector<pair<string, tensorflow::Tensor>> inputs = {{"input", outTensor}};
    auto uint8Caster = Cast(root.WithOpName("uint8_Cast"), outTensor, tensorflow::DT_UINT8);
	
    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output outTensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    vector<Tensor> outTensors;
    unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"uint8_Cast"}, {}, &outTensors));

    outTensor = outTensors.at(0);
    return Status::OK();

}

} //efficientdet

