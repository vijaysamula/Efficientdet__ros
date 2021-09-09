//STD
#include <unistd.h>
#include <string> 
#include <fstream>
// net stuff
#include <efficientdet_deploy/utils.h>
#include "efficientdet_deploy/handler.hpp"
//#include "tf_ros/NetTf.cpp"

namespace efficientdet {
/*!
 * Constructor.
 *
 * @param      nodeHandle  the ROS node handle.
 */

Handler::Handler(ros::NodeHandle& nodeHandle)
    : node_handle_(nodeHandle), it_(nodeHandle)
     {
    // Try to read the necessary parameters
    if (!readParameters()) {
    ROS_ERROR("Could not read parameters.");
    ros::requestShutdown();
    }
  loadLabelsFile();
    // Subscribe to images to infer
  ROS_INFO("Subscribing to image topic.");
  img_subscriber_ =
      it_.subscribe(input_image_topic_, 10, &Handler::imageCallback, this);

  // Advertise our topics
  ROS_INFO("Advertising our output.");
  // Advertise our topics
  //pred_publisher_ = it_.advertise(pred_image_topic_, 1);

  objectPublisher_ = node_handle_.advertise<darknet_ros_msgs::ObjectCount>("found_object", 1);
  boundingBoxesPublisher_ = node_handle_.advertise<darknet_ros_msgs::BoundingBoxes>("bounding_boxes", 1);
  detectionImagePublisher_ = it_.advertise("detection_image", 1);

  // Service servers.
  checkForObjectsServiceServer_ = node_handle_.advertiseService("check_for_objects", &Handler::checkForObjectsServiceCB, this);

  ROS_INFO("Successfully launched node.");
  

}

/*!
 * Destructor.
 */
Handler::~Handler() {}

/*!
 * @brief      Initialize the Handler
 *
 * @return     Error code
 */
void Handler::init() {
  // before doing anything, make sure we have a slash at the end of path
  modelPath_ += "/";
  //labelsMap_ = std::map<int,std::string>();
  try {
    net_ = std::unique_ptr<ModelLoader> (new ModelLoader(modelPath_ ,mem_percentage_,saved_model_));
  } catch (const std::invalid_argument& e) {
    std::cerr << "Unable to create network. " << std::endl
              << e.what() << std::endl;
    
  } catch (const std::runtime_error& e) {
    std::cerr << "Unable to init. network. " << std::endl
              << e.what() << std::endl;
   
  }

  
}

void Handler::loadLabelsFile()
{
  // get labels of all classes
  ifstream ifs(classesFile_.c_str());
  string line;
  while (getline(ifs, line)) labelsMap_.push_back(line);
  

}


void Handler::imageCallback(const sensor_msgs::ImageConstPtr& img_msg) {
  ros::Time begin = ros::Time::now();
  if (verbose_) {
    // report that we got something
    ROS_INFO("Image received.");
    ROS_INFO("Image encoding: %s", img_msg->encoding.c_str());
  }
  
  // Get the image
  cv_bridge::CvImageConstPtr cv_img;
  cv_img = cv_bridge::toCvShare(img_msg);
  
  darknet_ros_msgs::BoundingBoxes bb_msg;
  imagePrediction(cv_img->image,bb_msg);
  
  bb_msg.header.stamp = ros::Time::now();
  bb_msg.header.frame_id = "efficientdet_detections";
  bb_msg.image_header = img_msg->header;
  boundingBoxesPublisher_.publish(bb_msg);
  
  ros::Time now = ros::Time::now();
  auto diff_secs = (now.toSec()-begin.toSec());
  std::cout<<"The FPS is ----------------------------------------------------------------:"<<(diff_secs*1000 )<<"ms"<<std::endl;
}

bool Handler::checkForObjectsServiceCB(darknet_ros_msgs::CheckForObjects::Request &req, darknet_ros_msgs::CheckForObjects::Response &res
){
  ROS_DEBUG("[EfficientdetObjectDetector] Start check for objects service.");

  cv_bridge::CvImagePtr cam_image;
  try {
    cam_image = cv_bridge::toCvCopy(req.image, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return false;
  }

  
 
  darknet_ros_msgs::BoundingBoxes bb_msg;
  imagePrediction(cam_image->image,bb_msg);
  bb_msg.header.stamp = ros::Time::now();
  bb_msg.header.frame_id = "efficientdet_detections";
  bb_msg.image_header = req.image.header;

  darknet_ros_msgs::CheckForObjects::Response serviceResponse;
  serviceResponse.bounding_boxes = bb_msg;
  serviceResponse.id = req.id;
  res = serviceResponse;

  return true;
}

void Handler::imagePrediction(const cv::Mat& image ,darknet_ros_msgs::BoundingBoxes &bb_msg)
{
  // Infer with net
  std::vector<Tensor> predictions;
  net_->predict(image, predictions, thresholdIOU_,thresholdScore_,gray_);
  auto detections = predictions[0].flat_outer_dims<float,3>();
  
  std::vector<size_t> goodIdxs = filterBoxes(detections, thresholdIOU_, thresholdScore_);
  // Draw bboxes and captions
  //cvtColor(cv_img_bgr, cv_img_rgb, cv::COLOR_BGR2RGB);
  cv::Mat outImage = image.clone();
  bb_msg = drawBoundingBoxesOnImage(outImage, detections, labelsMap_, goodIdxs);
  //}
  //catch (const std::exception& e){
  //  cv_img_rgb = cv_img->image;
  //}
  if(show_opencv_) visualizeCVImage(outImage, "efficientdet_detections");
  if(publish_detection_image_) publishDetectionImage(outImage);
  // publishNumberOfDetections(numDetections(0));
  

}

void Handler::publishDetectionImage(const cv::Mat &cv_img)
{
  cv_bridge::CvImagePtr cv_ptr{new cv_bridge::CvImage};

  cv_ptr->header.frame_id = "efficientdet_detections";
  cv_ptr->header.stamp = ros::Time::now();
  cv_ptr->encoding = sensor_msgs::image_encodings::BGR8;
  cv_ptr->image = cv_img;

  sensor_msgs::Image::ConstPtr img_msg;
  try{
    img_msg = cv_ptr->toImageMsg();
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  detectionImagePublisher_.publish(img_msg);
}

bool Handler::readParameters() {
  if (!node_handle_.getParam("model_path", modelPath_) ||
      !node_handle_.getParam("memory_percentage", mem_percentage_)|| 
      !node_handle_.getParam("class_path", classesFile_) ||
      !node_handle_.getParam("saved_model", saved_model_) ||
      !node_handle_.getParam("threshold_iou", thresholdIOU_) ||
      !node_handle_.getParam("threshold_score", thresholdScore_) ||
      !node_handle_.getParam("show_opencv", show_opencv_) ||
      !node_handle_.getParam("publish_detection_image", publish_detection_image_) ||
      !node_handle_.getParam("gray", gray_) ||
      !node_handle_.getParam("input_image", input_image_topic_))
      
      
    return false;
  return true;
}
} // namespace efficientdet


