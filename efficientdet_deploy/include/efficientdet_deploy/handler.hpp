 #pragma once

 //ROS
 #include <cv_bridge/cv_bridge.h>
 #include <image_transport/image_transport.h>
 #include <ros/ros.h>
 #include <efficientdet_deploy/NetTf.hpp>
 //#include <tf_ros/NetTf.cpp>

 // efficientdet_ros_msgs
#include <efficientdet_ros_msgs/BoundingBox.h>
#include <efficientdet_ros_msgs/BoundingBoxes.h>
#include <efficientdet_ros_msgs/CheckForObjects.h>
#include <efficientdet_ros_msgs/ObjectCount.h>

 #include "tensorflow/core/framework/tensor.h"
#include <opencv2/imgcodecs.hpp> 
/* 
* Main class for the node to handle the ROS interfacing
*/
namespace efficientdet {
 class Handler {
    public:
     /*!
    * Constructor.
    *
    * @param      nodeHandle  the ROS node handle.
    */

    Handler (ros::NodeHandle& nodeHandle);

    /*!
   * Destructor.
   */
    virtual ~Handler();

    // /**
    // * @brief      Initialize the Handler
    // *
    // * @return     Error code
    // */
    // retCode init();

     /*!
    * loads the model path and model pb path.
    */
    void init();
    void loadLabelsFile();

    
    private:

    /*!
    * Reads and verifies the ROS parameters.
    *
    * @return     true if successful.
    */
    bool readParameters();


    /*!
    * ROS topic callback method.
    *
    * @param[in]  img_msg  The image message (to infer)
    */
    void imageCallback(const sensor_msgs::ImageConstPtr& img_msg);

    /*!
    * Check for objects service callback.
    */
    bool checkForObjectsServiceCB(efficientdet_ros_msgs::CheckForObjects::Request &req, efficientdet_ros_msgs::CheckForObjects::Response &res);
    
     /*!
    * predicts image
    * @param image 
    */
    void imagePrediction(const cv::Mat& image,efficientdet_ros_msgs::BoundingBoxes &bbmsg);
    
    /*!
    * publishes the image
    * @param image 
    */
    void publishDetectionImage(const cv::Mat &cv_img);
     //! ROS node handle.
    ros::NodeHandle& node_handle_;

    
    //! ROS topic subscribers and publishers.
    image_transport::ImageTransport it_;
    image_transport::Subscriber img_subscriber_;
    
    

    ros::Publisher objectPublisher_;
    ros::Publisher boundingBoxesPublisher_;
    image_transport::Publisher detectionImagePublisher_;
    
    //! Check for objects service server.
    ros::ServiceServer checkForObjectsServiceServer_;

    //! ROS topic names to subscribe to.
    std::string input_image_topic_;
    std::string pred_image_topic_;

    
    //! CNN related stuff
    std::unique_ptr<ModelLoader> net_;
    std::string modelPath_;
    bool verbose_;
    bool saved_model_;
    std::string device_;
    // Memory percentage to allocate the gpu
    float mem_percentage_;
    
    std::string classesFile_;
    std::vector<std::string> labelsMap_;

    double thresholdScore_;
    double thresholdIOU_;
 
    
    bool show_opencv_;
    
};
} // namespace efficientdet