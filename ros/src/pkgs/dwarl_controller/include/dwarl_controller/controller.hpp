// Nav2 controller plugin wrapping a DWARL policy installed with `dwarl deploy`.
#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include "dwarl_controller/policy_core.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_msgs/msg/collision_monitor_state.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace dwarl_controller
{

class DwarlController : public nav2_core::Controller
{
public:
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent, std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf, std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;
  void cleanup() override;
  void activate() override;
  void deactivate() override;
  void setPlan(const nav_msgs::msg::Path & path) override;
  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;
  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;
  void reset() override;

private:
  void onScan(sensor_msgs::msg::LaserScan::ConstSharedPtr msg);
  void onSafetyState(nav2_msgs::msg::CollisionMonitorState::ConstSharedPtr msg);
  void resetState();
  bool planPose(const geometry_msgs::msg::PoseStamped & pose, geometry_msgs::msg::PoseStamped & out) const;
  void addEvent(const geometry_msgs::msg::PoseStamped & pose, bool stop);
  void publishDebug(const rclcpp::Time & now, const std::string & status, bool stopped);
  void publishMetrics(double cycle_ms, double inference_ms, bool speed_limited, const std::string & status);

  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  rclcpp::Logger logger_{rclcpp::get_logger("DwarlController")};
  rclcpp::Clock::SharedPtr clock_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  std::string name_, base_frame_;
  DeploySpec spec_;
  std::unique_ptr<OnnxPolicy> policy_;
  PathTracker tracker_;
  nav_msgs::msg::Path plan_;
  std::array<double, 2> last_action_{0.0, 0.0}, cmd_{0.0, 0.0};
  std::vector<std::vector<float>> scan_buf_;
  double speed_limit_{0.0};  // absolute [m/s], 0 = none
  double scan_timeout_{0.5}, transform_tolerance_{0.2};
  bool have_goal_{false};
  geometry_msgs::msg::Point last_goal_;
  rclcpp::Time last_compute_{0, 0, RCL_ROS_TIME};

  std::mutex scan_mutex_;
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::PointStamped>::SharedPtr goal_pub_;
  rclcpp_lifecycle::LifecyclePublisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr metrics_pub_;

  // debug visualisation (`<name>.debug`): what the policy saw and did in the last control step
  struct DebugStep
  {
    std::vector<Vec2> goals;
    Vec2 projection;
    Pose2 pose;
    std::array<double, 2> action{0.0, 0.0}, velocity{0.0, 0.0};
    double dt{0.0}, scan_age{0.0};
  };
  bool debug_{false}, stop_logged_{false};
  bool profile_compute_{false};
  double min_linear_velocity_{0.0};
  bool reverse_limited_{false};
  size_t recorded_observations_{0};
  DebugStep dbg_;
  std::deque<std::pair<geometry_msgs::msg::Point, bool>> events_;  // (position, is safety stop)
  rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_pub_;

  // optional safety layer (Nav2 collision_monitor): a STOP fails FollowPath so the BT can back up and retry.
  // `safety_stop_` latches on STOP and is cleared when FollowPath restarts; the restarted policy then gets
  // `safety_grace_` s of commands so the monitor can re-evaluate (it still blocks them while unsafe).
  std::atomic<bool> safety_stop_{false};
  std::string safety_polygon_;
  double safety_grace_{0.3};
  rclcpp::Time session_start_{0, 0, RCL_ROS_TIME};
  rclcpp::Subscription<nav2_msgs::msg::CollisionMonitorState>::SharedPtr safety_sub_;
};

}  // namespace dwarl_controller
