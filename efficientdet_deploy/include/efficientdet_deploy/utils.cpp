#include "efficientdet_deploy/utils.h"

#include <math.h>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <regex>
#include <ros/ros.h>

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
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <cv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;



/** Draw bounding box and add caption to the image.
 *  Boolean flag _scaled_ shows if the passed coordinates are in relative units (true by default in tensorflow detection)
 */
void drawBoundingBoxOnImage(Mat &image, double yMin, double xMin, double yMax, double xMax, double score,  string label,bool scaled=false) {
    cv::Point tl, br;
    if (scaled) {
        tl = cv::Point((int) (xMin * image.cols), (int) (yMin * image.rows));
        br = cv::Point((int) (xMax * image.cols), (int) (yMax * image.rows));
    } else {
        tl = cv::Point((int) xMin, (int) yMin);
        br = cv::Point((int) xMax, (int) yMax);
    }
    cv::rectangle(image, tl, br, cv::Scalar(255, 255, 255), 10);

    // Ceiling the score down to 3 decimals (weird!)
    float scoreRounded = floorf(score * 1000) / 1000;
    string scoreString = to_string(scoreRounded).substr(0, 5);
    string caption = label + " (" + scoreString + ")";
    
    // Adding caption of type "LABEL (X.XXX)" to the top-left corner of the bounding box
    int fontCoeff = 24;
    cv::Point brRect = cv::Point(tl.x +  caption.length() *fontCoeff / 1.6, tl.y + fontCoeff);
    cv::rectangle(image, tl, brRect, cv::Scalar(0, 255, 255),4);
    cv::Point textCorner = cv::Point(tl.x, tl.y + fontCoeff * 0.9);
    cv::putText(image, caption, textCorner, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0));
}

/** Draw bounding boxes and add captions to the image.
 *  Box is drawn only if corresponding score is higher than the _threshold_.
 */
darknet_ros_msgs::BoundingBoxes drawBoundingBoxesOnImage(Mat &image,
                              tensorflow::TTypes<float>::Flat &scores,
                              tensorflow::TTypes<float>::Flat &classes,
                              tensorflow::TTypes<float,3>::Tensor &boxes,
                              vector<string>  &labelsMap, 
                              vector<size_t> &idxs) {
    darknet_ros_msgs::BoundingBoxes boundingBoxesResults;
    for (int j = 0; j < idxs.size(); j++) {
        darknet_ros_msgs::BoundingBox boundingBox;
        drawBoundingBoxOnImage(image,
                               boxes(0,idxs.at(j),0), boxes(0,idxs.at(j),1),
                               boxes(0,idxs.at(j),2), boxes(0,idxs.at(j),3),
                               scores(idxs.at(j)),labelsMap[classes(idxs.at(j))]
                               );
        double ymin = boxes(0,idxs.at(j),0);
        double xmin = boxes(0,idxs.at(j),1);
        double ymax = boxes(0,idxs.at(j),2);
        double xmax = boxes(0,idxs.at(j),3);

        boundingBox.Class = labelsMap[classes(idxs.at(j))];
        boundingBox.id = static_cast<int8_t>(j);
        boundingBox.probability = scores(idxs.at(j));
        boundingBox.xmin = xmin;
        boundingBox.ymin = ymin;
        boundingBox.xmax = xmax;
        boundingBox.ymax = ymax;
        boundingBoxesResults.bounding_boxes.push_back(boundingBox);
    }
    
    return boundingBoxesResults;
}

darknet_ros_msgs::BoundingBoxes drawBoundingBoxesOnImage(Mat &image,
                              tensorflow::TTypes<float, 3>::Tensor &detections,
                              vector<string>  &labelsMap, 
                              vector<size_t> &idxs) {
    darknet_ros_msgs::BoundingBoxes boundingBoxesResults;
    
    for (int j = 0; j < idxs.size(); j++) {
        darknet_ros_msgs::BoundingBox boundingBox;
        float classLabel = detections(0,idxs.at(j),6);
        if (classLabel >(labelsMap.size()-1))
            classLabel =0;
        if (labelsMap[classLabel]!="person") continue;
        
        drawBoundingBoxOnImage(image,
                               detections(0,idxs.at(j),1), detections(0,idxs.at(j),2),
                               detections(0,idxs.at(j),3), detections(0,idxs.at(j),4),
                               detections(0,idxs.at(j),5),labelsMap[classLabel]
                               );
        auto ymin = detections(0,idxs.at(j),1);
        auto xmin = detections(0,idxs.at(j),2);
        auto ymax = detections(0,idxs.at(j),3);
        auto xmax = detections(0,idxs.at(j),4);

        boundingBox.Class = labelsMap[classLabel];
        boundingBox.id = static_cast<int8_t>(j);
        boundingBox.probability = detections(0,idxs.at(j),5);
        boundingBox.xmin = xmin;
        boundingBox.ymin = ymin;
        boundingBox.xmax = xmax;
        boundingBox.ymax = ymax;
        boundingBoxesResults.bounding_boxes.push_back(boundingBox);
    }
    
    return boundingBoxesResults;
}

/** Calculate intersection-over-union (IOU) for two given bbox Rects.
 */
double IOU(Rect2f box1, Rect2f box2) {

    float xA = max(box1.tl().x, box2.tl().x);
    float yA = max(box1.tl().y, box2.tl().y);
    float xB = min(box1.br().x, box2.br().x);
    float yB = min(box1.br().y, box2.br().y);

    float intersectArea = abs((xB - xA) * (yB - yA));
    float unionArea = abs(box1.area()) + abs(box2.area()) - intersectArea;

    return 1. * intersectArea / unionArea;
}

/** Return idxs of good boxes (ones with highest confidence score (>= thresholdScore)
 *  and IOU <= thresholdIOU with others).
 */
vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
                           tensorflow::TTypes<float, 3>::Tensor &boxes, 
                           double thresholdIOU, double thresholdScore) {

    

    vector<size_t> sortIdxs(scores.size());
    iota(sortIdxs.begin(), sortIdxs.end(), 0);
    
    // Create set of "bad" idxs
    set<size_t> badIdxs = set<size_t>();
    size_t i = 0;
    while (i < sortIdxs.size()) {
        if (scores(sortIdxs.at(i)) < thresholdScore)
            badIdxs.insert(sortIdxs[i]);
        if (badIdxs.find(sortIdxs.at(i)) != badIdxs.end()) {
            i++;
            continue;
        }

        
        
        Rect2f box1 = Rect2f(Point2f(boxes(0, sortIdxs.at(i), 1), boxes(0, sortIdxs.at(i), 0)),
                             Point2f(boxes(0, sortIdxs.at(i), 3), boxes(0, sortIdxs.at(i), 2)));
        for (size_t j = i + 1; j < sortIdxs.size(); j++) {
            if (scores(sortIdxs.at(j)) < thresholdScore) {
                badIdxs.insert(sortIdxs[j]);
                continue;
            }
            Rect2f box2 = Rect2f(Point2f(boxes(0, sortIdxs.at(j), 1), boxes(0, sortIdxs.at(j), 0)),
                                 Point2f(boxes(0, sortIdxs.at(j), 3), boxes(0, sortIdxs.at(j), 2)));
            if (IOU(box1, box2) > thresholdIOU)
                badIdxs.insert(sortIdxs[j]);
        }
        i++;
    }
    
    // Prepare "good" idxs for return
    vector<size_t> goodIdxs = vector<size_t>();
    for (auto it = sortIdxs.begin(); it != sortIdxs.end(); it++)
        if (badIdxs.find(sortIdxs.at(*it)) == badIdxs.end())
            goodIdxs.push_back(*it);

    return goodIdxs;
}

vector<size_t> filterBoxes(tensorflow::TTypes<float, 3>::Tensor &detections,
                           double thresholdIOU, double thresholdScore) {

    

    vector<size_t> sortIdxs(detections.dimension(1));
    iota(sortIdxs.begin(), sortIdxs.end(), 0);
    
    // Create set of "bad" idxs
    set<size_t> badIdxs = set<size_t>();
    size_t i = 0;
    while (i < sortIdxs.size()) {
        float score1 = detections(0,sortIdxs.at(i),5);
        float label1 = detections(0,sortIdxs.at(i),6);
        if ( score1< thresholdScore) {
            badIdxs.insert(sortIdxs[i]);
         
        }
        if (badIdxs.find(sortIdxs.at(i)) != badIdxs.end()) {
            i++;
            continue;
        }
        
        // Rect2f box1 = Rect2f(Point2f(detections(0, sortIdxs.at(i), 2), detections(0, sortIdxs.at(i), 1)),
        //                      Point2f(detections(0, sortIdxs.at(i), 4), detections(0, sortIdxs.at(i), 3)));
        // for (size_t j = i + 1; j < sortIdxs.size(); j++) {
        //     float score2 = detections(0,sortIdxs.at(j),5);
        //     float label2 = detections(0,sortIdxs.at(j),6);
        //     if (label2 == label1) {
        //     if (score2 < thresholdScore) {
        //         badIdxs.insert(sortIdxs[j]);
        //         continue;
        //     }
        //     Rect2f box2 = Rect2f(Point2f(detections(0, sortIdxs.at(j), 2), detections(0, sortIdxs.at(j), 1)),
        //                      Point2f(detections(0, sortIdxs.at(j), 4), detections(0, sortIdxs.at(j), 3)));
        //     if (IOU(box1, box2) > thresholdIOU)
        //         badIdxs.insert(sortIdxs[j]);
        //     }
        // }
        i++;
    }
    
    // Prepare "good" idxs for return
    vector<size_t> goodIdxs = vector<size_t>();
    for (auto it = sortIdxs.begin(); it != sortIdxs.end(); it++)
        if (badIdxs.find(sortIdxs.at(*it)) == badIdxs.end())
            goodIdxs.push_back(*it);
    
    return goodIdxs;
}
void visualizeCVImage(const cv::Mat &img, const std::string &window_name){

  cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
  cv::imshow( window_name, img );
  cv::waitKey(1);
}

