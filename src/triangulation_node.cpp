
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
#include "yolo_triangulation/visualization_functions.h"

// Service types
#include <mg_msgs/set_strings.h>
#include <std_srvs/Trigger.h>

class Triangulation
{
  ros::NodeHandle nh_;
  std::vector<ros::Subscriber> pose_sub_, bbox_sub_;
  std::string in_pose_topic_, in_BBox_topic_;
  std::vector<std::string> tracking_class_names_;
  nav_msgs::Odometry odom_;
  std::vector<Eigen::MatrixXd> TransfMat_;
  Eigen::Matrix3d CamMatrix_;
  Eigen::Quaterniond q_enu2cam_;
  Eigen::Vector3d cam_pos_body_frame_;
  std::vector<std::string> namespaces_;

  std::vector<std::vector<Eigen::Vector2d>> feature_points_;
  std::vector<std::vector<Eigen::MatrixXd>> ProjMatrices_;

  ros::Publisher detection_rviz_pub_;
  ros::ServiceServer start_new_batch_srv_;
  ros::ServiceServer stop_batch_srv_;
  bool get_data_;  // Boolean to start/stop collecting yolo data

public:
  Triangulation(ros::NodeHandle *nh) {

    nh_ = *nh;
    // Get parameters
    nh_.getParam("input_pose_topic", in_pose_topic_);
    nh_.getParam("input_bounding_boxes_topic", in_BBox_topic_);
    nh_.getParam("namespaces", namespaces_);
    // nh_.getParam("tracking_class_name", tracking_class_name_);

    // Get cam position in body frame
    std::vector<double> cam_pos;
    nh_.getParam("cam_pos_body_frame", cam_pos);
    cam_pos_body_frame_ = Eigen::Vector3d(cam_pos[0], cam_pos[1], cam_pos[2]);
    // std::cout << "camera position in body frame: " << cam_pos_body_frame_ << std::endl;

    // Camera parameters
    double fx, fy, cx, cy;
    nh_.getParam("fx", fx);
    nh_.getParam("fy", fy);
    nh_.getParam("cx", cx);
    nh_.getParam("cy", cy);
    this->SetCamMatrix(fx, fy, cx, cy);
    // std::cout << "CamMatrix_: " << std::endl << CamMatrix_ << std::endl;

    // Quaternion to rotate from z pointing up to z pointing outwards from camera
    q_enu2cam_ = Eigen::Quaterniond(0.5, -0.5, 0.5, -0.5);

    // Set callback to capture data for triangulation
    get_data_ = false;

    // Publisher to display emoji location in Rviz
    detection_rviz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("fault_detection", 2);

    // Services to start/stop collecting data for triangulation
    start_new_batch_srv_ = nh_.advertiseService("collect_yolo_data", &Triangulation::CollectData, this);
    stop_batch_srv_ = nh_.advertiseService("stop_collecting_yolo_data", &Triangulation::StopDataCollection, this);

    // Subscribe to input video feed and publish output video feed
    pose_sub_.resize(namespaces_.size());
    bbox_sub_.resize(namespaces_.size());
    TransfMat_.resize(namespaces_.size());
    for (uint i = 0; i < namespaces_.size(); i++) {
      std::string pose_topic = "/" + namespaces_[i] + in_pose_topic_;
      std::string BBox_topic = "/" + namespaces_[i] + in_BBox_topic_;
      pose_sub_[i] = nh_.subscribe<nav_msgs::Odometry>
          (pose_topic, 1, boost::bind(&Triangulation::pose_callback, this, _1, i));
      bbox_sub_[i] = nh_.subscribe<darknet_ros_msgs::BoundingBoxes>
          (BBox_topic, 1, boost::bind(&Triangulation::yolo_callback, this, _1, i));
      ROS_INFO("[YOLO triangulation] Input pose topic: %s", pose_sub_[i].getTopic().c_str());
      ROS_INFO("[YOLO triangulation] Input bounding box topic: %s", bbox_sub_[i].getTopic().c_str());

      // Set matrix size for TransfMat
      TransfMat_[i] = Eigen::MatrixXd(3,4);
    }
    // ROS_INFO("[YOLO triangulation] Triangulating class: %s", tracking_class_name_.c_str());

  }

  ~Triangulation() {
  }

  void SetCamMatrix(const double &fx, const double &fy, const double &cx, const double &cy) {
    CamMatrix_ << fx,  0.0,  cx,
                  0.0,  fy,  cy,
                  0.0, 0.0, 1.0;
  }

  // Index represents the source of pose when swarming
  void pose_callback(const nav_msgs::Odometry::ConstPtr &odom, const uint &index) {
    odom_ = *odom;

    // Camera rotation
    Eigen::Quaterniond q_enu = msg_conversions::ros_to_eigen_quat(odom_.pose.pose.orientation);
    Eigen::Quaterniond q_cam = q_enu*q_enu2cam_;
    Eigen::Matrix3d Rwc = q_cam.toRotationMatrix();
    Eigen::Matrix3d Rcw = Rwc.transpose();

    // Camera position
    Eigen::Vector3d pos_w = 
      q_enu.toRotationMatrix()*cam_pos_body_frame_ + msg_conversions::ros_point_to_eigen_vector(odom_.pose.pose.position);
    Eigen::Vector3d pos_c = -Rcw*pos_w;

    TransfMat_[index] << Rcw(0,0), Rcw(0,1), Rcw(0,2), pos_c(0),
                         Rcw(1,0), Rcw(1,1), Rcw(1,2), pos_c(1),
                         Rcw(2,0), Rcw(2,1), Rcw(2,2), pos_c(2);
  }

  // Index represents the source of the detection when swarming
  void yolo_callback(const darknet_ros_msgs::BoundingBoxes::ConstPtr& bbox, const uint &index) {
    if (!get_data_) {
      return;
    }
    
    if(odom_.header.stamp.toSec() == 0) {
      ROS_WARN("[YOLO triangulation] No pose data!");
      return;
    }

    for(uint i = 0; i < bbox->bounding_boxes.size(); i++) {
      for (uint j = 0; j < tracking_class_names_.size(); j++) {
        if (bbox->bounding_boxes[i].Class == tracking_class_names_[j]) {
          double x_center = 0.5*(bbox->bounding_boxes[i].xmin + bbox->bounding_boxes[i].xmax);
          double y_center = 0.5*(bbox->bounding_boxes[i].ymin + bbox->bounding_boxes[i].ymax);
          feature_points_[j].push_back(Eigen::Vector2d(x_center, y_center));
          ProjMatrices_[j].push_back(CamMatrix_*TransfMat_[index]);
          this->SolveTriangulation(false);
        }
      }
    }
    // ROS_INFO("[YOLO triangulation] Number of feature points: %zd", feature_points_.size());

  }

  bool CollectData(mg_msgs::set_strings::Request &req,
                   mg_msgs::set_strings::Response &res) {
    if (req.strings.size() == 0) {
      get_data_ = false;
      res.success = false;
    } else {
      get_data_ = true;
      res.success = true;
      tracking_class_names_ = req.strings;
      feature_points_.clear();
      ProjMatrices_.clear();
      for (uint j = 0; j < tracking_class_names_.size(); j++) {
        std::vector<Eigen::Vector2d> feature_vec;
        std::vector<Eigen::MatrixXd> proj_matrices;
        feature_points_.push_back(feature_vec);
        ProjMatrices_.push_back(proj_matrices);
        ROS_INFO("[YOLO triangulation] Starting data collection for class %s", tracking_class_names_[j].c_str());
      }
    }
    return true;
  }

  bool StopDataCollection(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
    get_data_ = false;

    this->SolveTriangulation(true);

    res.success = true;
    return true;
  }

  void SolveTriangulation(const bool &print_info) {
    // Rviz variables
    std::vector<Eigen::Vector3d> feature_positions;
    std::vector<std::string> feature_names;

    for (uint j = 0; j < tracking_class_names_.size(); j++) {
      if (feature_points_[j].size() < 3) {
        // ROS_INFO("[YOLO triangulation] Not enough feature points for class %s", tracking_class_names_[j].c_str());
        continue;
      }

      Eigen::Vector3d feature_pos;
      this->triangulateNViews(feature_points_[j], ProjMatrices_[j], &feature_pos);

      if(print_info) {
        ROS_INFO("[YOLO triangulation] Calculating feature location for class '%s' with %zd measurements.", 
                tracking_class_names_[j].c_str(), feature_points_[j].size());
        std::cout << tracking_class_names_[j] << " position = " << std::endl << feature_pos << std::endl;
      }
      feature_positions.push_back(feature_pos);
      feature_names.push_back(tracking_class_names_[j]);
    }

    this->PublishDefectsRviz(feature_positions, feature_names);
  }

  void PublishDefectsRviz(const std::vector<Eigen::Vector3d> &feature_positions, 
                          const std::vector<std::string> &feature_names) {
    const std::string frame_id = "map", ns = "defect_position";
    const double size = 0.05;
    const std_msgs::ColorRGBA orange_color = visualization_functions::Color::Orange();
    const std_msgs::ColorRGBA yellow_color = visualization_functions::Color::Yellow();
    const double transparency = 1.0;
    visualization_msgs::MarkerArray marker_array;

    // Create markers for positions
    visualization_functions::DrawNodes(feature_positions, frame_id, ns, size,
               orange_color, transparency, &marker_array);

    // Name the markers
    for (uint j = 0; j < feature_names.size(); j++) {
      visualization_functions::NameMarker(feature_positions[j], feature_names[j], frame_id,
               feature_names[j], yellow_color, 0, &marker_array);
    }

    // Publish
    detection_rviz_pub_.publish(marker_array);
  }

  void triangulateNViews(const std::vector<Eigen::Vector2d> &feature_points,
                         const std::vector<Eigen::MatrixXd> &proj_matrices,
                         Eigen::Vector3d *feature_pos) {
    uint n_points = feature_points.size();
    Eigen::MatrixXd M(2*n_points, 4);

    for (uint i = 0; i < n_points; i++) {
      double x = feature_points[i](0);
      double y = feature_points[i](1);
      Eigen::Vector4d p1 = proj_matrices[i].row(0);
      Eigen::Vector4d p2 = proj_matrices[i].row(1);
      Eigen::Vector4d p3 = proj_matrices[i].row(2);
      M.row(i*2 + 0) = x*p3 - p1;
      M.row(i*2 + 1) = y*p3 - p2;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::Vector4d X_h = V.col(3);
    *feature_pos = Eigen::Vector3d(X_h(0)/X_h(3), X_h(1)/X_h(3), X_h(2)/X_h(3));
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