#include "dwarl_controller/controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "nav2_core/controller_exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace dwarl_controller
{

using nav2_util::declare_parameter_if_not_declared;

void DwarlController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent, std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf, std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  auto node = parent.lock();
  if (!node) {
    throw nav2_core::ControllerException("unable to lock node");
  }
  name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;
  logger_ = node->get_logger();
  clock_ = node->get_clock();
  base_frame_ = costmap_ros->getBaseFrameID();

  declare_parameter_if_not_declared(node, name + ".model_dir", rclcpp::ParameterValue(""));
  declare_parameter_if_not_declared(node, name + ".scan_topic", rclcpp::ParameterValue("scan"));
  declare_parameter_if_not_declared(node, name + ".scan_timeout", rclcpp::ParameterValue(0.5));
  declare_parameter_if_not_declared(node, name + ".transform_tolerance", rclcpp::ParameterValue(0.2));
  declare_parameter_if_not_declared(node, name + ".inference_threads", rclcpp::ParameterValue(1));
  declare_parameter_if_not_declared(node, name + ".safety_state_topic", rclcpp::ParameterValue(""));
  declare_parameter_if_not_declared(node, name + ".safety_grace", rclcpp::ParameterValue(0.3));
  declare_parameter_if_not_declared(node, name + ".debug", rclcpp::ParameterValue(false));
  // refuse a bundle trained for another robot ("" = accept any)
  declare_parameter_if_not_declared(node, name + ".expected_robot_id", rclcpp::ParameterValue(""));

  const auto dir = node->get_parameter(name + ".model_dir").as_string();
  if (dir.empty()) {throw nav2_core::ControllerException("model_dir must name an explicit policy bundle");}
  spec_ = DeploySpec::load(dir);
  const auto expected = node->get_parameter(name + ".expected_robot_id").as_string();
  if (!expected.empty() && expected != spec_.robot_id) {
    throw nav2_core::ControllerException(
            "policy bundle " + dir + " was trained for robot '" + spec_.robot_id + "', expected '" + expected + "'");
  }
  policy_ = std::make_unique<OnnxPolicy>(spec_, node->get_parameter(name + ".inference_threads").as_int());
  scan_timeout_ = node->get_parameter(name + ".scan_timeout").as_double();
  transform_tolerance_ = node->get_parameter(name + ".transform_tolerance").as_double();

  scan_sub_ = node->create_subscription<sensor_msgs::msg::LaserScan>(
    node->get_parameter(name + ".scan_topic").as_string(), rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::LaserScan::ConstSharedPtr msg) {onScan(msg);});
  goal_pub_ = node->create_publisher<geometry_msgs::msg::PointStamped>(name + "/lookahead", 1);
  metrics_pub_ = node->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(name + "/metrics", 10);
  debug_ = node->get_parameter(name + ".debug").as_bool();
  debug_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(name + "/debug_markers", 1);
  safety_grace_ = node->get_parameter(name + ".safety_grace").as_double();
  const auto safety_topic = node->get_parameter(name + ".safety_state_topic").as_string();
  if (!safety_topic.empty()) {
    safety_sub_ = node->create_subscription<nav2_msgs::msg::CollisionMonitorState>(
      safety_topic, rclcpp::QoS(10),
      [this](nav2_msgs::msg::CollisionMonitorState::ConstSharedPtr msg) {onSafetyState(msg);});
    RCLCPP_INFO(logger_, "safety layer: a collision_monitor STOP on '%s' fails FollowPath", safety_topic.c_str());
  }

  double freq = 0.0;
  node->get_parameter("controller_frequency", freq);
  if (freq > 0.0 && std::abs(1.0 / freq - spec_.dt) > 1e-3) {
    RCLCPP_WARN(logger_, "controller_frequency %.1f Hz differs from the training rate %.1f Hz", freq, 1.0 / spec_.dt);
  }
  RCLCPP_INFO(
    logger_, "DWARL policy %s for robot '%s' (goal %s %.1f m, v [%.2f, %.2f] m/s, w %.2f rad/s)",
    spec_.onnx_file.c_str(), spec_.robot_id.c_str(), spec_.goal_mode.c_str(), spec_.lookahead_distance, spec_.v_min,
    spec_.v_max, spec_.w_max);
}

void DwarlController::cleanup()
{
  scan_sub_.reset();
  safety_sub_.reset();
  goal_pub_.reset();
  metrics_pub_.reset();
  debug_pub_.reset();
  costmap_ros_.reset();
  policy_.reset();
}

void DwarlController::activate()
{
  goal_pub_->on_activate();
  metrics_pub_->on_activate();
  debug_pub_->on_activate();
  resetState();
}

void DwarlController::deactivate()
{
  goal_pub_->on_deactivate();
  metrics_pub_->on_deactivate();
  debug_pub_->on_deactivate();
}

void DwarlController::resetState()
{
  last_action_ = {0.0, 0.0};
  cmd_ = {0.0, 0.0};
  scan_buf_.clear();
  have_goal_ = false;
  safety_stop_ = false;
}

void DwarlController::reset() {resetState();}

void DwarlController::onScan(sensor_msgs::msg::LaserScan::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lk(scan_mutex_);
  scan_ = msg;
}

void DwarlController::onSafetyState(nav2_msgs::msg::CollisionMonitorState::ConstSharedPtr msg)
{
  // latched: once stopped the monitor sees zero commands and reports "continue" again, which must not
  // release the policy; only a FollowPath restart (new session, after the BT's back-up) clears it
  if (msg->action_type != nav2_msgs::msg::CollisionMonitorState::STOP) {return;}
  {
    std::lock_guard<std::mutex> lk(scan_mutex_);
    safety_polygon_ = msg->polygon_name;
  }
  safety_stop_ = true;
}

void DwarlController::setPlan(const nav_msgs::msg::Path & path)
{
  if (path.poses.empty()) {
    throw nav2_core::InvalidPath("received an empty plan");
  }
  const auto & g = path.poses.back().pose.position;
  // a new goal starts a new "episode" (training resets action/command at episode start)
  const bool new_goal = !have_goal_ ||
    std::hypot(g.x - last_goal_.x, g.y - last_goal_.y) > spec_.goal_tolerance ||
    path.header.frame_id != plan_.header.frame_id;
  if (new_goal) {
    last_action_ = {0.0, 0.0};
    cmd_ = {0.0, 0.0};
    scan_buf_.clear();
  }
  have_goal_ = true;
  last_goal_ = g;
  plan_ = path;
  std::vector<Vec2> pts;
  pts.reserve(path.poses.size());
  for (const auto & p : path.poses) {
    if (!std::isfinite(p.pose.position.x) || !std::isfinite(p.pose.position.y)) {
      throw nav2_core::InvalidPath("nonfinite plan");
    }
    pts.push_back({p.pose.position.x, p.pose.position.y});
  }
  tracker_.setPlan(pts);
}

geometry_msgs::msg::TwistStamped DwarlController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker *)
{
  const auto cycle_start = std::chrono::steady_clock::now();
  const auto elapsed = [&]() {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cycle_start).count();
    };
  if (tracker_.empty()) {
    throw nav2_core::InvalidPath("no plan");
  }
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan;
  {
    std::lock_guard<std::mutex> lk(scan_mutex_);
    scan = scan_;
  }
  const rclcpp::Time now = clock_->now();
  // a gap in control calls == FollowPath (re)started: the robot was stopped, so start like a training episode
  const bool new_session = last_compute_.nanoseconds() == 0 || (now - last_compute_).seconds() > 3.0 * spec_.dt;
  if (new_session) {
    session_start_ = now;
    safety_stop_ = false;
    last_action_ = {0.0, 0.0};
    cmd_ = {0.0, 0.0};
    scan_buf_.clear();
    stop_logged_ = false;
  }
  const bool debug = debug_ && debug_pub_->is_activated();
  if (debug && new_session) {
    geometry_msgs::msg::PoseStamped p;
    if (planPose(pose, p)) {addEvent(p, false);}
  }
  if (safety_sub_ && safety_stop_ && (now - session_start_).seconds() > safety_grace_) {
    std::string poly;
    {
      std::lock_guard<std::mutex> lk(scan_mutex_);
      poly = safety_polygon_;
    }
    last_compute_ = now;  // controller_server keeps calling within failure_tolerance: same session
    cmd_ = {0.0, 0.0};
    publishMetrics(elapsed(), 0., false, "safety_stop");
    if (debug) {
      geometry_msgs::msg::PoseStamped p;
      if (planPose(pose, p)) {
        if (!stop_logged_) {addEvent(p, true);}
        dbg_.pose = {p.pose.position.x, p.pose.position.y, tf2::getYaw(p.pose.orientation)};
      }
      stop_logged_ = true;
      dbg_.velocity = {velocity.linear.x, velocity.angular.z};
      publishDebug(now, "SAFETY STOP (" + poly + ")", true);
    }
    RCLCPP_WARN_THROTTLE(
      logger_, *clock_, 1000, "safety stop by collision_monitor (%s): failing FollowPath for recovery", poly.c_str());
    throw nav2_core::NoValidControl("safety stop by collision_monitor (" + poly + ")");
  }
  if (!scan || (now - rclcpp::Time(scan->header.stamp, now.get_clock_type())).seconds() > scan_timeout_) {
    cmd_ = {0., 0.};
    publishMetrics(elapsed(), 0., false, "stale_scan");
    throw nav2_core::NoValidControl("no recent laser scan");
  }

  // robot pose in the plan frame (usually map), laser orientation w.r.t. the base frame
  geometry_msgs::msg::PoseStamped p;
  geometry_msgs::msg::TransformStamped laser_tf;
  try {
    p = tf_->transform(pose, plan_.header.frame_id, tf2::durationFromSec(transform_tolerance_));
    laser_tf = tf_->lookupTransform(base_frame_, scan->header.frame_id, tf2::TimePointZero);
  } catch (const tf2::TransformException & e) {
    throw nav2_core::ControllerTFError(e.what());
  }
  const auto & lt = laser_tf.transform;  // full 3D mount (e.g. an upside-down lidar mirrors the scan)
  const LaserPose laser = LaserPose::fromQuaternion(
    lt.translation.x, lt.translation.y, lt.translation.z, lt.rotation.x, lt.rotation.y, lt.rotation.z,
    lt.rotation.w);
  if (scan->range_max > 0.0 && scan->range_max < spec_.max_range) {
    RCLCPP_WARN_ONCE(
      logger_, "laser range_max %.2f m < training max_range %.2f m: obstacles beyond %.2f m look like free space",
      scan->range_max, spec_.max_range, scan->range_max);
  }

  Scan s;
  s.angle_min = scan->angle_min;
  s.angle_increment = scan->angle_increment;
  s.range_min = scan->range_min;
  s.range_max = scan->range_max;
  s.ranges = scan->ranges;
  const auto scan_lidar = lidarObservation(spec_, s, laser);
  const size_t depth = static_cast<size_t>(spec_.lidar_history.back() + 1);
  if (scan_buf_.empty()) {
    scan_buf_.assign(depth, scan_lidar);
  } else {
    scan_buf_.insert(scan_buf_.begin(), scan_lidar);
    scan_buf_.resize(depth);
  }
  std::vector<float> lidar;
  lidar.reserve(spec_.lidar_history.size() * spec_.num_rays);
  for (const int lag : spec_.lidar_history) {
    lidar.insert(lidar.end(), scan_buf_[lag].begin(), scan_buf_[lag].end());
  }
  Vec2 goal;
  const Pose2 pose2{p.pose.position.x, p.pose.position.y, tf2::getYaw(p.pose.orientation)};
  const auto state = stateObservation(spec_, tracker_, pose2, velocity.linear.x, velocity.angular.z, last_action_,
      cmd_, &goal, debug ? &dbg_.goals : nullptr, debug ? &dbg_.projection : nullptr);
  if (!std::all_of(state.begin(), state.end(), [](float v) {return std::isfinite(v);})) {
    cmd_ = {0., 0.};
    publishMetrics(elapsed(), 0., false, "nonfinite_observation");
    throw nav2_core::NoValidControl("nonfinite observation");
  }

  const auto inference_start = std::chrono::steady_clock::now();
  std::array<double, 2> a = (*policy_)(state, lidar);
  const double inference_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - inference_start).count();
  if (!std::isfinite(a[0]) || !std::isfinite(a[1])) {
    cmd_ = {0., 0.};
    publishMetrics(elapsed(), inference_ms, false, "nonfinite_policy_output");
    throw nav2_core::NoValidControl("nonfinite policy output");
  }
  a = {std::clamp(a[0], -1.0, 1.0), std::clamp(a[1], -1.0, 1.0)};
  cmd_ = commandFromAction(spec_, a, cmd_);
  last_action_ = a;
  const bool speed_limited = speed_limit_ > 0. && std::abs(cmd_[0]) > speed_limit_;

  const double step_dt = new_session ? 0.0 : (now - last_compute_).seconds();
  if (!new_session) {
    const double dt = (now - last_compute_).seconds();
    if (std::abs(dt - spec_.dt) > 0.5 * spec_.dt) {
      RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000, "control period %.3f s, policy trained at %.3f s", dt, spec_.dt);
    }
  }
  last_compute_ = now;

  if (goal_pub_->is_activated()) {
    geometry_msgs::msg::PointStamped gp;
    gp.header = plan_.header;
    gp.header.stamp = now;
    gp.point.x = goal.x;
    gp.point.y = goal.y;
    goal_pub_->publish(gp);
  }
  if (debug) {
    dbg_.pose = pose2;
    dbg_.action = a;
    dbg_.velocity = {velocity.linear.x, velocity.angular.z};
    dbg_.dt = step_dt;
    dbg_.scan_age = (now - rclcpp::Time(scan->header.stamp, now.get_clock_type())).seconds();
    const bool grace = safety_stop_ && (now - session_start_).seconds() <= safety_grace_;
    publishDebug(now, new_session ? "NEW SESSION" : grace ? "GRACE" : "RUNNING", false);
  }

  geometry_msgs::msg::TwistStamped out;
  out.header.frame_id = base_frame_;
  out.header.stamp = now;
  out.twist.linear.x = speed_limit_ > 0.0 ? std::clamp(cmd_[0], -speed_limit_, speed_limit_) : cmd_[0];
  out.twist.angular.z = cmd_[1];
  publishMetrics(elapsed(), inference_ms, speed_limited, "running");
  return out;
}

void DwarlController::publishMetrics(
  double cycle_ms, double inference_ms, bool speed_limited, const std::string & status)
{
  if (!metrics_pub_ || !metrics_pub_->is_activated()) {return;}
  diagnostic_msgs::msg::DiagnosticArray message;
  message.header.stamp = clock_->now();
  diagnostic_msgs::msg::DiagnosticStatus d;
  d.name = name_ + "/policy";
  d.hardware_id = spec_.robot_id;
  d.message = status;
  d.level = status == "running" ? 0 : 1;
  const auto add = [&](const std::string & key, const std::string & value) {
      diagnostic_msgs::msg::KeyValue kv;
      kv.key = key;
      kv.value = value;
      d.values.push_back(kv);
    };
  add("cycle_ms_excluding_metrics_publish", std::to_string(cycle_ms));
  add("inference_ms", std::to_string(inference_ms));
  add("speed_limit_intervention", speed_limited ? "1" : "0");
  add("safety_stop", status == "safety_stop" ? "1" : "0");
  add("fallback_active", "0");  // plugin itself never invokes a classical fallback
  add("command_v", std::to_string(cmd_[0]));
  add("command_w", std::to_string(cmd_[1]));
  add("policy_action_v", std::to_string(last_action_[0]));
  add("policy_action_w", std::to_string(last_action_[1]));
  message.status.push_back(d);
  metrics_pub_->publish(message);
}

bool DwarlController::planPose(
  const geometry_msgs::msg::PoseStamped & pose, geometry_msgs::msg::PoseStamped & out) const
{
  try {
    out = tf_->transform(pose, plan_.header.frame_id, tf2::durationFromSec(transform_tolerance_));
  } catch (const tf2::TransformException &) {
    return false;
  }
  return true;
}

void DwarlController::addEvent(const geometry_msgs::msg::PoseStamped & pose, bool stop)
{
  events_.emplace_back(pose.pose.position, stop);
  while (events_.size() > 50) {events_.pop_front();}
}

void DwarlController::publishDebug(const rclcpp::Time & now, const std::string & status, bool stopped)
{
  using visualization_msgs::msg::Marker;
  auto point = [](double x, double y, double z = 0.0) {
      geometry_msgs::msg::Point p;
      p.x = x;
      p.y = y;
      p.z = z;
      return p;
    };
  auto color = [](float r, float g, float b, float a = 1.0f) {
      std_msgs::msg::ColorRGBA c;
      c.r = r;
      c.g = g;
      c.b = b;
      c.a = a;
      return c;
    };
  auto marker = [&](const std::string & ns, int id, int type, bool in_map) {
      Marker m;
      m.header.frame_id = in_map ? plan_.header.frame_id : base_frame_;
      m.header.stamp = now;
      m.ns = ns;
      m.id = id;
      m.type = type;
      m.action = Marker::ADD;
      m.pose.orientation.w = 1.0;
      m.lifetime = rclcpp::Duration::from_seconds(stopped ? 2.0 : 5.0 * spec_.dt);
      return m;
    };
  visualization_msgs::msg::MarkerArray arr;
  const auto & d = dbg_;

  if (!stopped) {
    // goals: near (green) -> far (red), with rays from the robot
    Marker goals = marker("goals", 0, Marker::SPHERE_LIST, true);
    Marker rays = marker("goals", 1, Marker::LINE_LIST, true);
    goals.scale.x = goals.scale.y = goals.scale.z = 0.1;
    rays.scale.x = 0.01;
    const size_t n = d.goals.size();
    for (size_t i = 0; i < n; ++i) {
      const float t = n > 1 ? static_cast<float>(i) / static_cast<float>(n - 1) : 0.0f;
      const auto c = color(t, 1.0f - t, 0.0f);
      goals.points.push_back(point(d.goals[i].x, d.goals[i].y, 0.05));
      goals.colors.push_back(c);
      rays.points.push_back(point(d.pose.x, d.pose.y, 0.05));
      rays.points.push_back(point(d.goals[i].x, d.goals[i].y, 0.05));
      rays.colors.push_back(c);
      rays.colors.push_back(c);
    }
    arr.markers.push_back(goals);
    arr.markers.push_back(rays);

    // policy_scan: newest preprocessed scan decoded back to metres, from the training lidar origin
    if (!scan_buf_.empty()) {
      Marker scan = marker("policy_scan", 0, Marker::POINTS, false);
      scan.scale.x = scan.scale.y = 0.04;
      scan.color = color(1.0f, 0.0f, 1.0f);
      const auto & obs = scan_buf_[0];
      for (size_t i = 0; i < obs.size(); ++i) {
        if (obs[i] >= 0.999f) {continue;}  // no return
        const double r = lidarRange(spec_, obs[i]), th = spec_.angle_min + i * spec_.angle_increment;
        scan.points.push_back(point(spec_.origin_x + r * std::cos(th), spec_.origin_y + r * std::sin(th), 0.1));
      }
      arr.markers.push_back(scan);
    }

    // tracked_path: projection window around the current segment and the projection point
    const auto & pts = tracker_.points();
    if (!pts.empty()) {
      Marker path = marker("tracked_path", 0, Marker::LINE_STRIP, true);
      path.scale.x = 0.03;
      path.color = color(0.0f, 0.8f, 1.0f, 0.8f);
      const int last = static_cast<int>(pts.size()) - 1;
      const int lo = std::clamp(tracker_.segment() + spec_.window_lo, 0, last);
      const int hi = std::clamp(tracker_.segment() + spec_.window_hi + 1, 0, last);
      for (int i = lo; i <= hi; ++i) {path.points.push_back(point(pts[i].x, pts[i].y, 0.02));}
      Marker proj = marker("tracked_path", 1, Marker::SPHERE, true);
      proj.pose.position = point(d.projection.x, d.projection.y, 0.02);
      proj.scale.x = proj.scale.y = proj.scale.z = 0.08;
      proj.color = color(0.0f, 0.8f, 1.0f);
      arr.markers.push_back(path);
      arr.markers.push_back(proj);
    }
  }

  // cmd: (v, w) as an arrow to (v, w * v_max / w_max); raw action target, rate-limited command, measured
  const std::array<double, 2> raw = spec_.action_mode == "window" ?
    std::array<double, 2>{cmd_[0], cmd_[1]} :
    std::array<double, 2>{spec_.v_min + (d.action[0] + 1.0) * 0.5 * (spec_.v_max - spec_.v_min),
    d.action[1] * spec_.w_max};
  const double w_to_y = spec_.v_max / spec_.w_max;
  const std::array<std::pair<std::array<double, 2>, std_msgs::msg::ColorRGBA>, 3> arrows{{
    {raw, color(1.0f, 0.6f, 0.0f)}, {cmd_, color(0.0f, 1.0f, 0.0f)}, {d.velocity, color(0.2f, 0.4f, 1.0f)}}};
  for (int i = 0; i < 3; ++i) {
    Marker m = marker("cmd", i, Marker::ARROW, false);
    m.points = {point(0.0, 0.0, 0.3), point(arrows[i].first[0], arrows[i].first[1] * w_to_y, 0.3)};
    m.scale.x = 0.02;
    m.scale.y = 0.05;
    m.scale.z = 0.05;
    m.color = arrows[i].second;
    arr.markers.push_back(m);
  }

  Marker text = marker("status", 0, Marker::TEXT_VIEW_FACING, false);
  text.pose.position = point(0.0, 0.0, 0.8);
  text.scale.z = 0.12;
  text.color = stopped ? color(1.0f, 0.2f, 0.2f) : color(1.0f, 1.0f, 1.0f);
  char buf[160];
  std::snprintf(
    buf, sizeof(buf), "%s\ncmd v %.2f w %.2f | meas v %.2f w %.2f\ndt %.3f s  scan age %.3f s", status.c_str(),
    cmd_[0], cmd_[1], d.velocity[0], d.velocity[1], d.dt, d.scan_age);
  text.text = buf;
  arr.markers.push_back(text);

  // events: persistent history of safety stops (red) and new sessions (blue)
  for (int stop = 0; stop < 2; ++stop) {
    Marker ev = marker("events", stop, Marker::SPHERE_LIST, true);
    ev.lifetime = rclcpp::Duration::from_seconds(0.0);
    ev.scale.x = ev.scale.y = ev.scale.z = stop ? 0.12 : 0.08;
    ev.color = stop ? color(1.0f, 0.0f, 0.0f, 0.8f) : color(0.0f, 0.3f, 1.0f, 0.8f);
    for (const auto & e : events_) {
      if (e.second == static_cast<bool>(stop)) {ev.points.push_back(point(e.first.x, e.first.y, 0.15));}
    }
    if (ev.points.empty()) {ev.action = Marker::DELETE;}
    arr.markers.push_back(ev);
  }

  Marker fp = marker("events", 2, Marker::LINE_STRIP, false);
  if (stopped && costmap_ros_) {
    fp.scale.x = 0.04;
    fp.color = color(1.0f, 0.0f, 0.0f);
    for (const auto & q : costmap_ros_->getRobotFootprint()) {fp.points.push_back(point(q.x, q.y, 0.05));}
    if (!fp.points.empty()) {fp.points.push_back(fp.points.front());}
  } else {
    fp.action = Marker::DELETE;
  }
  arr.markers.push_back(fp);
  debug_pub_->publish(arr);
}

void DwarlController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  if (speed_limit <= 0.0 || (percentage && speed_limit >= 100.0)) {
    speed_limit_ = 0.0;  // nav2_costmap_2d::NO_SPEED_LIMIT
  } else {
    speed_limit_ = percentage ? spec_.v_max * speed_limit / 100.0 : speed_limit;
  }
}

}  // namespace dwarl_controller

PLUGINLIB_EXPORT_CLASS(dwarl_controller::DwarlController, nav2_core::Controller)
