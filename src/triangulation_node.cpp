
// ROS
#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>

// Opencv
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>

// Eigen
#include <Eigen/Dense>

// darknet_ros_msgs
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <darknet_ros_msgs/BoundingBox.h>

// Locally defined libraries
#include "yolo_triangulation/msg_conversions.h"
#include "yolo_triangulation/triangulation.hpp"
uint count = 0;
class Triangulation
{
  ros::NodeHandle nh_;
  ros::Subscriber pose_sub_, bbox_sub_;
  std::string in_pose_topic_, in_BBox_topic_, tracking_class_name_;
  nav_msgs::Odometry odom_;
  cv::Matx34d TransfMat_;
  cv::Matx33d CamMatrix_;
  Eigen::Quaterniond q_enu2cam_;

  // Vectors for calculating triangulation
  std::vector<cv::Point2d> feature_points_;
  std::vector<cv::Matx34d> ProjMatrices_;

public:
  Triangulation(ros::NodeHandle *nh) {

    nh_ = *nh;
    // Get parameters
    nh_.getParam("input_pose_topic", in_pose_topic_);
    nh_.getParam("input_bounding_boxes_topic", in_BBox_topic_);
    nh_.getParam("tracking_class_name", tracking_class_name_);
    
    // Camera parameters
    double fx, fy, cx, cy;
    nh_.getParam("fx", fx);
    nh_.getParam("fy", fy);
    nh_.getParam("cx", cx);
    nh_.getParam("cy", cy);
    this->SetCamMatrix(fx, fy, cx, cy);
    std::cout << "CamMatrix_: " << std::endl << CamMatrix_ << std::endl;

    // Quaternion to rotate from z pointing up to z pointing outwards from camera
    q_enu2cam_ = Eigen::Quaterniond(0.5, -0.5, 0.5, -0.5);

    // Subscribe to input video feed and publish output video feed
    pose_sub_ = nh_.subscribe(in_pose_topic_, 1, &Triangulation::pose_callback, this);
    bbox_sub_ = nh_.subscribe(in_BBox_topic_, 1, &Triangulation::yolo_callback, this);

    ROS_INFO("[YOLO triangulation] Input pose topic: %s", pose_sub_.getTopic().c_str());
    ROS_INFO("[YOLO triangulation] Input bounding box topic: %s", bbox_sub_.getTopic().c_str());
    ROS_INFO("[YOLO triangulation] Triangulating class: %s", tracking_class_name_.c_str());
    // ROS_INFO("Output topic: %s", image_pub_.getTopic().c_str());

  }

  ~Triangulation() {
  }

  void SetCamMatrix(const double &fx, const double &fy, const double &cx, const double &cy) {
    CamMatrix_ = cv::Matx33d(fx,  0.0,  cx,
                             0.0,  fy,  cy,
                             0.0, 0.0, 1.0);
  }

  void pose_callback(const nav_msgs::Odometry::ConstPtr &odom) {
    odom_ = *odom;

    // Camera rotation
    Eigen::Quaterniond q_enu = msg_conversions::ros_to_eigen_quat(odom_.pose.pose.orientation);
    Eigen::Quaterniond q_cam = q_enu*q_enu2cam_;
    Eigen::Matrix3d Rcw = q_cam.toRotationMatrix().transpose();

    // Camera position
    Eigen::Vector3d pos_w = msg_conversions::ros_point_to_eigen_vector(odom_.pose.pose.position);
    Eigen::Vector3d pos_c = -Rcw*pos_w;

    TransfMat_ = cv::Matx34d(Rcw(0,0), Rcw(0,1), Rcw(0,2), pos_c(0),
                             Rcw(1,0), Rcw(1,1), Rcw(1,2), pos_c(1),
                             Rcw(2,0), Rcw(2,1), Rcw(2,2), pos_c(2));
  }

  void yolo_callback(const darknet_ros_msgs::BoundingBoxes& bbox) {
    if(odom_.header.stamp.toSec() == 0) {
      ROS_WARN("[YOLO triangulation] No pose data!");
      return;
    }
    // std::cout << "dt: " << (bbox.header.stamp - odom_.header.stamp).toSec() << std::endl;

    for(uint i = 0; i < bbox.bounding_boxes.size(); i++) {
      if (bbox.bounding_boxes[i].Class == tracking_class_name_) {
       count++;
       double x_center = 0.5*(bbox.bounding_boxes[i].xmin + bbox.bounding_boxes[i].xmax);
       double y_center = 0.5*(bbox.bounding_boxes[i].ymin + bbox.bounding_boxes[i].ymax);
       feature_points_.push_back(cv::Point2d(x_center, y_center));
       ProjMatrices_.push_back(CamMatrix_*TransfMat_);
       // std::cout << "p(" << count << ",:) = " << cv::Point2d(x_center, y_center) << ";" << std::endl;
       // std::cout << "T(:,:," << count << ") = " << TransfMat_ << ";" << std::endl;
       // std::cout << "CamMatrix_: " << std::endl << CamMatrix_ << std::endl;
      }
    }
    // ROS_INFO("Number of feature points: %zd", feature_points_.size());

    if(feature_points_.size() > 10) {
      // ROS_INFO("Calculating feature location!");
      uint N = feature_points_.size();
      cv::Mat_<double> points(2, N);
      for (uint i = 0; i < feature_points_.size(); i++) {
        points.at<double>(0,i) = feature_points_[i].x;
        points.at<double>(1,i) = feature_points_[i].y;
      }
      cv::Vec3d X;
      cv::sfm::triangulateNViews(points, ProjMatrices_, X);
      // cv::Vec4d X_h(X[0], X[1], X[2], 1.0);
      // ROS_INFO("Feature location calculated: ");
      // std::cout << "solution point: " << X << std::endl;
      // for (uint i = 0; i < feature_points_.size(); i++) {
        // cv::Vec3d proj_point = ProjMatrices_[i]*X_h;
        // std::cout << "p_proj(" << i+1 <<  ",:) = " << proj_point/proj_point[2] << ";" << std::endl;
        // std::cout << "measured point: " << feature_points_[i] << std::endl;
      //   std::cout << "error: [" << proj_point[0]/proj_point[2] - feature_points_[i].x <<
      //                ", " << proj_point[1]/proj_point[2] - feature_points_[i].y << std::endl;
      // }
    }
  }
};


int main(int argc, char **argv)
{
  ros::init(argc, argv, "yolo_triangulation");
  ros::NodeHandle node("~");
  Triangulation ic(&node);

  ros::spin();

  return 0;
}