/* Copyright (c) 2017, United States Government, as represented by the
 * Administrator of the National Aeronautics and Space Administration.
 * 
 * All rights reserved.
 * 
 * The Astrobee platform is licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include <string>
#include <vector>
#include <set>
#include "yolo_triangulation/visualization_functions.h"

namespace visualization_functions {

// Overwrites the given properties of the marker array.
void SetMarkerProperties(const std_msgs::Header &header,
                         const double &life_time,
                         visualization_msgs::MarkerArray* markers) {
  int count = 0;
  for (visualization_msgs::Marker& marker : markers->markers) {
    marker.header = header;
    marker.id = count;
    marker.lifetime = ros::Duration(life_time);
    ++count;
  }
}

void DrawNodes(const std::vector<Eigen::Vector3d> &points,
               const std::string &frame_id,
               const std::string &ns,  // namespace
               const double &size,
               const std_msgs::ColorRGBA &color,
               const double &transparency,  // 0 -> transparent, 1 -> opaque
               visualization_msgs::MarkerArray *marker_array) {
  // marker_array->markers.clear();

  visualization_msgs::Marker marker;
  marker.type = visualization_msgs::Marker::CUBE_LIST;
  marker.color = color;
  marker.color.a = transparency;
  marker.scale.x = size;
  marker.scale.y = size;
  marker.scale.z = size;
  marker.ns = ns;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time::now();
  marker.pose.orientation.w = 1.0;
  marker.header.seq = 0;
  marker.id = 0;

  // Get the number of requested waypoints
  uint n_w = points.size();

  if (n_w == 0) {
    marker.action = visualization_msgs::Marker::DELETE;
  } else {
    marker.action = visualization_msgs::Marker::ADD;
  }

  for (size_t i = 0; i < n_w; ++i) {
    geometry_msgs::Point NewPoint;
    NewPoint.x = points[i](0);
    NewPoint.y = points[i](1);
    NewPoint.z = points[i](2);
    marker.points.push_back(NewPoint);
    // marker.pose.position = NewPoint;
    // marker_array->markers.push_back(marker);
    // i = i + 1;
  }
  marker_array->markers.push_back(marker);

  std_msgs::Header header;
  header.frame_id = frame_id;
  header.stamp = ros::Time::now();
  SetMarkerProperties(header, 0.0, marker_array);
}

void NameMarker(const Eigen::Vector3d &point,
                const std::string &text,
                const std::string &frame_id,
                const std::string &ns,  // namespace
                const std_msgs::ColorRGBA &color,
                const int &seqNumber,
                visualization_msgs::MarkerArray *markerArray) {
  visualization_msgs::Marker marker;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::Marker::ADD;
  marker.color = color;
  marker.text = text;
  marker.color.a = 1.0;
  marker.scale.z = 0.05;
  marker.ns = ns;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time::now();
  marker.pose.position = 
    msg_conversions::set_ros_point(point(0), point(1), point(2)+0.15);
  marker.pose.orientation.w = 1.0;
  marker.id = seqNumber;
  marker.lifetime = ros::Duration(0.0);

  markerArray->markers.push_back(marker);
}

}  // namespace visualization_functions
