// ROS-free core of the DWARL controller: reproduces the training observations/actions from deploy.yaml.
// Mirrors dwarl/core/envs/nav_env.py (_compute_obs, _track_path, step) and dwarl/core/sim/lidar.py.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace dwarl_controller
{

struct DeploySpec
{
  std::string dir;  // directory containing deploy.yaml
  std::string onnx_file;
  std::string obs_input{"obs"}, lidar_input{"lidar"}, action_output{"actions"};
  int obs_dim{7};
  double dt{0.1};
  // lidar: ray i starts at (origin_x, origin_y) in the base frame along bearing angle_min + i * angle_increment
  int num_rays{180};
  double angle_min{0.0}, angle_increment{0.0}, min_range{0.12}, max_range{8.0};
  double origin_x{0.0}, origin_y{0.0};
  bool full_circle{true};
  std::string lidar_normalize{"linear"};
  std::vector<int> lidar_history{0};
  std::string robot_id;
  // state
  double max_goal_distance{10.0}, v_scale{0.8}, w_scale{1.5};
  // goal
  std::string goal_mode{"lookahead"};
  double lookahead_distance{2.0}, goal_tolerance{0.3};
  std::vector<double> lookahead_distances{2.0};
  int window_lo{-5}, window_hi{30};
  // action
  double v_min{-0.2}, v_max{0.8}, w_max{1.5}, a_max{1.0}, alpha_max{3.0};
  std::string action_mode{"absolute"};
  bool command_feedback{false};

  static constexpr int kVersion = 5;
  static DeploySpec load(const std::string & dir);  // throws std::runtime_error
};

struct Vec2
{
  double x{0.0}, y{0.0};
};

struct Scan
{
  double angle_min{0.0}, angle_increment{0.0}, range_min{0.0}, range_max{0.0};
  std::vector<float> ranges;
};

// Laser mounting pose in the robot (base) frame: full 3D rotation (row-major) + translation, so tilted or
// upside-down mounts are handled (a beam at laser angle th points along R * (cos th, sin th, 0)).
struct LaserPose
{
  std::array<double, 9> R{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::array<double, 3> t{0, 0, 0};
  static LaserPose planar(double x, double y, double yaw);
  // from a unit quaternion (x, y, z, w) and translation
  static LaserPose fromQuaternion(double tx, double ty, double tz, double qx, double qy, double qz, double qw);
};

// Resample an arbitrary planar scan onto the training ray layout and normalise to [0, 1].
// Training casts rays from the lidar origin (spec.origin_x/y in the base frame), so each return is re-projected
// into the base frame and binned by its bearing from that origin; a bin keeps the closest return.
// No-return readings (nan, +inf, < range_min incl. 0, > range_max) -> max_range; -inf -> hit at range_min.
// Bins that no beam falls into (scan coarser than training) copy the nearest covered bin.
std::vector<float> lidarObservation(const DeploySpec & spec, const Scan & scan, const LaserPose & laser = {});

// Projects the robot onto the global plan like training does (windowed search around the last segment).
class PathTracker
{
public:
  void setPlan(const std::vector<Vec2> & plan);
  bool empty() const {return pts_.empty();}
  // arc length of the projection (updates the segment index); first call after setPlan searches globally
  double project(const Vec2 & xy, int window_lo, int window_hi);
  Vec2 pointAt(double s) const;
  Vec2 goal() const {return pts_.back();}
  double length() const {return s_.empty() ? 0.0 : s_.back();}
  const std::vector<Vec2> & points() const {return pts_;}
  int segment() const {return seg_;}

private:
  std::vector<Vec2> pts_;
  std::vector<double> s_;
  int seg_{0};
  bool fresh_{true};
};

struct Pose2
{
  double x{0.0}, y{0.0}, yaw{0.0};
};

// Goal features [min(d,gmax)/gmax, sin, cos] for every configured lookahead, then action and velocity; returns viz
// goal, every goal point fed to the policy and the projection point on the path.
std::vector<float> stateObservation(
  const DeploySpec & spec, PathTracker & tracker, const Pose2 & pose, double v, double w,
  const std::array<double, 2> & last_action, const std::array<double, 2> & command,
  Vec2 * goal_out = nullptr, std::vector<Vec2> * goals_out = nullptr, Vec2 * projection_out = nullptr);

// Inverse of the lidar normalisation: observation value in [0, 1] -> range [m] from the lidar origin.
double lidarRange(const DeploySpec & spec, float value);

// action in [-1, 1]^2 -> rate-limited (v, w) command, exactly as the training env.
std::array<double, 2> commandFromAction(
  const DeploySpec & spec, const std::array<double, 2> & action, const std::array<double, 2> & cmd);

class OnnxPolicy
{
public:
  explicit OnnxPolicy(const DeploySpec & spec, int threads = 1);
  ~OnnxPolicy();
  std::array<double, 2> operator()(const std::vector<float> & state, const std::vector<float> & lidar);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Resolve a model directory: explicit `model_dir`, else <root>/runs/train/<run>/exported ("latest" = newest
// run with an exported/deploy.yaml). root defaults to $DWARL_ROOT. Throws std::runtime_error.

}  // namespace dwarl_controller
