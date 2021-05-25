#ifndef TF_DETECTOR_EXAMPLE_UTILS_H
#define TF_DETECTOR_EXAMPLE_UTILS_H

#endif //TF_DETECTOR_EXAMPLE_UTILS_H

#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include <opencv2/core/mat.hpp>

// eficientdet_ros_msgs
#include <efficientdet_ros_msgs/BoundingBox.h>
#include <efficientdet_ros_msgs/BoundingBoxes.h>

using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;


Status readLabelsMapFile(const string &fileName, std::map<int, string> &labelsMap);

Status loadGraph(const string &graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session);

Status readTensorFromMat(const cv::Mat &mat, Tensor &outTensor);

void drawBoundingBoxOnImage(cv::Mat &image, double xMin, double yMin, double xMax, double yMax, double score,string label, bool scaled);

efficientdet_ros_msgs::BoundingBoxes drawBoundingBoxesOnImage(cv::Mat &image,
                              tensorflow::TTypes<float>::Flat &scores,
                              tensorflow::TTypes<float>::Flat &classes,
                              tensorflow::TTypes<float,3>::Tensor &boxes,
                              std::vector<std::string>  &labelsMap,
                              std::vector<size_t> &idxs);

efficientdet_ros_msgs::BoundingBoxes drawBoundingBoxesOnImage(cv::Mat &image,
                              tensorflow::TTypes<float, 3>::Tensor &detections,
                              std::vector<std::string>  &labelsMap,
                              std::vector<size_t> &idxs);

double IOU(cv::Rect box1, cv::Rect box2);

std::vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
                                tensorflow::TTypes<float, 3>::Tensor &boxes,
                                double thresholdIOU, double thresholdScore);

std::vector<size_t> filterBoxes(tensorflow::TTypes<float, 3>::Tensor &detections,
                                double thresholdIOU, double thresholdScore);
void visualizeCVImage(const cv::Mat &img, const std::string &window_name);
