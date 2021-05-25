#include <ros/ros.h>
#include "efficientdet_deploy/handler.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "efficientdet_deploy");
  ros::NodeHandle nodeHandle("~");



  efficientdet::Handler efficientdetHandler( nodeHandle);
  efficientdetHandler.init();

  ros::spin();
  return 0;
}