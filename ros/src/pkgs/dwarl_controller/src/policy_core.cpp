#include "dwarl_controller/policy_core.hpp"

#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;

namespace dwarl_controller
{

namespace
{
double wrap(double a) {return std::remainder(a, 2.0 * M_PI);}
}  // namespace

DeploySpec DeploySpec::load(const std::string & dir)
{
  const auto file = fs::path(dir) / "deploy.yaml";
  if (!fs::exists(file)) {
    throw std::runtime_error("deploy.yaml not found in " + dir);
  }
  const YAML::Node y = YAML::LoadFile(file.string());
  const int version = y["version"].as<int>();
  if (version != 2 && version != 3 && version != kVersion) {
    throw std::runtime_error(
            "deploy.yaml version " + y["version"].as<std::string>() + " in " + dir + ", expected " +
            "2, 3 or " + std::to_string(kVersion) + " (prepare a compatible bundle with `dwarl deploy`)");
  }
  DeploySpec s;
  s.dir = dir;
  s.onnx_file = (fs::path(dir) / y["model"]["onnx"].as<std::string>()).string();
  for (const auto & in : y["model"]["inputs"]) {
    const auto group = in["group"].as<std::string>();
    if (group == "state") {
      s.obs_input = in["name"].as<std::string>();
      s.obs_dim = in["shape"][1].as<int>();
    } else if (group == "lidar") {
      s.lidar_input = in["name"].as<std::string>();
    }
  }
  s.action_output = y["model"]["outputs"][0]["name"].as<std::string>();
  s.dt = y["control"]["dt"].as<double>();
  const auto l = y["lidar"];
  s.num_rays = l["num_rays"].as<int>();
  s.angle_min = l["angle_min"].as<double>();
  s.angle_increment = l["angle_increment"].as<double>();
  s.min_range = l["min_range"].as<double>();
  s.max_range = l["max_range"].as<double>();
  s.origin_x = l["origin"][0].as<double>();
  s.origin_y = l["origin"][1].as<double>();
  s.full_circle = l["full_circle"].as<bool>();
  s.lidar_normalize = version >= 3 && l["normalize"] ? l["normalize"].as<std::string>() : "linear";
  if (version >= 3 && l["history"]) {
    s.lidar_history.clear();
    for (const auto & k : l["history"]) {s.lidar_history.push_back(k.as<int>());}
  }
  s.robot_id = y["robot"]["id"] ? y["robot"]["id"].as<std::string>() : "";
  const auto st = y["state"];
  s.command_feedback = st["command_feedback"] && st["command_feedback"].as<bool>();
  if (s.command_feedback != (version == 5)) {
    throw std::runtime_error("command feedback requires deploy version 5");
  }
  s.max_goal_distance = st["max_goal_distance"].as<double>();
  s.v_scale = st["v_scale"].as<double>();
  s.w_scale = st["w_scale"].as<double>();
  const auto g = y["goal"];
  if (version == 5 && (!g["tracker"] || g["tracker"].as<std::string>() != "segment_window_v1")) {
    throw std::runtime_error("v5 requires segment_window_v1 guidance");
  }
  s.goal_mode = g["mode"].as<std::string>();
  s.lookahead_distance = g["lookahead_distance"].as<double>();
  if (version >= 3 && g["lookahead_distances"]) {
    s.lookahead_distances.clear();
    for (const auto & d : g["lookahead_distances"]) {s.lookahead_distances.push_back(d.as<double>());}
  }
  s.goal_tolerance = g["tolerance"].as<double>();
  s.window_lo = g["projection_window"][0].as<int>();
  s.window_hi = g["projection_window"][1].as<int>();
  const auto a = y["action"];
  s.v_min = a["v"]["min"].as<double>();
  s.v_max = a["v"]["max"].as<double>();
  s.w_max = a["w"]["max"].as<double>();
  s.a_max = a["rate_limit"]["a_max"].as<double>();
  s.alpha_max = a["rate_limit"]["alpha_max"].as<double>();
  s.action_mode = version >= 3 && a["mode"] ? a["mode"].as<std::string>() : "absolute";
  if (y["robot"]["drive_type"].as<std::string>() != "diff") {
    throw std::runtime_error("only diff-drive policies are supported");
  }
  if (s.lidar_normalize != "linear" && s.lidar_normalize != "log") {
    throw std::runtime_error("unknown lidar normalization");
  }
  if (s.lidar_history.empty() || s.lidar_history.front() != 0 || !std::is_sorted(
      s.lidar_history.begin(), s.lidar_history.end())) {
    throw std::runtime_error("lidar history must be ascending and start with 0");
  }
  if (s.action_mode != "absolute" && s.action_mode != "window") {
    throw std::runtime_error("unknown action mode");
  }
  if (s.obs_dim != static_cast<int>(3 * s.lookahead_distances.size() + 4 + 2 * s.command_feedback)) {
    throw std::runtime_error("state observation dimension does not match lookahead distances");
  }
  return s;
}

LaserPose LaserPose::planar(double x, double y, double yaw)
{
  LaserPose p;
  const double c = std::cos(yaw), s = std::sin(yaw);
  p.R = {c, -s, 0, s, c, 0, 0, 0, 1};
  p.t = {x, y, 0};
  return p;
}

LaserPose LaserPose::fromQuaternion(
  double tx, double ty, double tz, double qx, double qy, double qz, double qw)
{
  LaserPose p;
  p.R = {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
    2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
    2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)};
  p.t = {tx, ty, tz};
  return p;
}

std::vector<float> lidarObservation(const DeploySpec & spec, const Scan & scan, const LaserPose & laser)
{
  const int nr = spec.num_rays;
  const double inf = std::numeric_limits<double>::infinity();
  const bool train_full = spec.full_circle;
  // bin index of a bearing in the base frame, -1 if outside the training FOV
  auto bin = [&](double bearing) -> long {
      const double rel = wrap(bearing - spec.angle_min - M_PI) + M_PI;  // [0, 2pi)
      const long k = std::lround(rel / spec.angle_increment);
      if (train_full) {return k % nr;}
      return k < nr ? k : -1;
    };

  std::vector<double> best(nr, inf);  // closest hit per bin (inf = covered but no return)
  std::vector<char> seen(nr, 0);
  const int n = static_cast<int>(scan.ranges.size());
  const auto & R = laser.R;
  const double ox = laser.t[0] - spec.origin_x, oy = laser.t[1] - spec.origin_y;  // laser rel. to ray origin
  for (int j = 0; j < n; ++j) {
    const double v = scan.ranges[j];
    const double th = scan.angle_min + j * scan.angle_increment;  // laser frame
    const double ct = std::cos(th), st = std::sin(th);
    const double ux = R[0] * ct + R[1] * st, uy = R[3] * ct + R[4] * st;  // beam direction, base xy
    double r;
    if (v == -inf) {
      r = scan.range_min;  // REP 117: too close to measure
    } else if (std::isnan(v) || v == inf || v < scan.range_min || (scan.range_max > 0.0 && v > scan.range_max)) {
      r = inf;  // no return
    } else {
      r = v;
    }
    // for no-returns, bin along a far point so the bearing is the ray direction as seen from the ray origin
    const double d = std::isfinite(r) ? r : std::max(spec.max_range, 1.0) * 10.0;
    const double px = ox + d * ux, py = oy + d * uy;
    const long k = bin(std::atan2(py, px));
    if (k < 0) {continue;}
    seen[k] = 1;
    if (std::isfinite(r)) {
      best[k] = std::min(best[k], std::hypot(px, py));
    }
  }

  // fill bins no beam landed in from the nearest covered bin (bounded by the scan's own resolution)
  const int reach = std::max(1, static_cast<int>(std::ceil(
      std::abs(scan.angle_increment) / spec.angle_increment)));
  std::vector<double> filled = best;
  for (int i = 0; i < nr; ++i) {
    if (seen[i]) {continue;}
    for (int o = 1; o <= reach; ++o) {
      int lo = i - o, hi = i + o;
      if (train_full) {
        lo = (lo + nr) % nr;
        hi = hi % nr;
      }
      const bool l = lo >= 0 && lo < nr && seen[lo], h = hi >= 0 && hi < nr && seen[hi];
      if (l || h) {
        filled[i] = std::min(l ? best[lo] : inf, h ? best[hi] : inf);
        break;
      }
    }
  }

  std::vector<float> out(nr);
  const double span = spec.max_range - spec.min_range;
  for (int i = 0; i < nr; ++i) {
    const double r = std::clamp(std::isfinite(filled[i]) ? filled[i] : spec.max_range, spec.min_range,
        spec.max_range);
    out[i] = static_cast<float>(spec.lidar_normalize == "log" ?
        std::log1p(std::max(r - spec.min_range, 0.0) / 0.1) / std::log1p(span / 0.1) :
        (r - spec.min_range) / span);
  }
  return out;
}

double lidarRange(const DeploySpec & spec, float value)
{
  const double span = spec.max_range - spec.min_range;
  return spec.min_range + (spec.lidar_normalize == "log" ? 0.1 * std::expm1(value * std::log1p(span / 0.1)) :
         value * span);
}

void PathTracker::setPlan(const std::vector<Vec2> & plan)
{
  if (plan.empty()) {throw std::runtime_error("empty plan");}
  for (const auto & p : plan) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y)) {throw std::runtime_error("nonfinite plan");}
  }
  pts_ = plan;
  s_.assign(pts_.size(), 0.0);
  for (size_t i = 1; i < pts_.size(); ++i) {
    s_[i] = s_[i - 1] + std::hypot(pts_[i].x - pts_[i - 1].x, pts_[i].y - pts_[i - 1].y);
  }
  seg_ = 0;
  fresh_ = true;
}

double PathTracker::project(const Vec2 & xy, int window_lo, int window_hi)
{
  if (!std::isfinite(xy.x) || !std::isfinite(xy.y)) {throw std::runtime_error("nonfinite position");}
  if (pts_.size() < 2) {
    return 0.0;
  }
  const int last = static_cast<int>(pts_.size()) - 2;
  int lo = fresh_ ? 0 : seg_ + window_lo;
  int hi = fresh_ ? last + 1 : seg_ + window_hi;  // exclusive
  fresh_ = false;
  double best = std::numeric_limits<double>::infinity(), best_s = 0.0;
  int best_seg = seg_;
  for (int o = lo; o < hi; ++o) {
    const int c = std::clamp(o, 0, last);
    const Vec2 & a = pts_[c], & b = pts_[c + 1];
    const double abx = b.x - a.x, aby = b.y - a.y;
    const double l2 = std::max(abx * abx + aby * aby, 1e-12);
    const double t = std::clamp(((xy.x - a.x) * abx + (xy.y - a.y) * aby) / l2, 0.0, 1.0);
    const double d = std::hypot(xy.x - (a.x + t * abx), xy.y - (a.y + t * aby));
    if (d < best) {
      best = d;
      best_seg = c;
      best_s = s_[c] + t * std::sqrt(l2);
    }
  }
  seg_ = best_seg;
  return best_s;
}

Vec2 PathTracker::pointAt(double s) const
{
  // np.interp semantics: clamp to the ends, linear in between
  if (s <= s_.front()) {return pts_.front();}
  if (s >= s_.back()) {return pts_.back();}
  const auto it = std::upper_bound(s_.begin(), s_.end(), s);
  const size_t i = static_cast<size_t>(it - s_.begin());  // s_[i-1] <= s < s_[i]
  const double ds = s_[i] - s_[i - 1];
  const double t = ds > 0 ? (s - s_[i - 1]) / ds : 0.0;
  return {pts_[i - 1].x + t * (pts_[i].x - pts_[i - 1].x), pts_[i - 1].y + t * (pts_[i].y - pts_[i - 1].y)};
}

std::vector<float> stateObservation(
  const DeploySpec & spec, PathTracker & tracker, const Pose2 & pose, double v, double w,
  const std::array<double, 2> & last_action, const std::array<double, 2> & command,
  Vec2 * goal_out, std::vector<Vec2> * goals_out, Vec2 * projection_out)
{
  const double s = tracker.project({pose.x, pose.y}, spec.window_lo, spec.window_hi);
  if (projection_out) {*projection_out = tracker.pointAt(s);}
  if (goals_out) {goals_out->clear();}
  const Vec2 goal = spec.goal_mode == "lookahead" ? tracker.pointAt(s + spec.lookahead_distance) : tracker.goal();
  if (goal_out) {*goal_out = goal;}
  std::vector<float> out;
  const double c = std::cos(pose.yaw), sn = std::sin(pose.yaw), gmax = spec.max_goal_distance;
  const auto goals = spec.goal_mode == "lookahead" ? spec.lookahead_distances : std::vector<double>{0.0};
  for (const double d : goals) {
    const Vec2 point = spec.goal_mode == "lookahead" ? tracker.pointAt(s + d) : goal;
    if (goals_out) {goals_out->push_back(point);}
    const double dx = point.x - pose.x, dy = point.y - pose.y;
    const double gx = c * dx + sn * dy, gy = -sn * dx + c * dy;
    const double dist = std::hypot(gx, gy), ang = std::atan2(gy, gx);
    out.insert(out.end(), {static_cast<float>(std::min(dist, gmax) / gmax), static_cast<float>(std::sin(ang)),
      static_cast<float>(std::cos(ang))});
  }
  out.insert(out.end(), {static_cast<float>(last_action[0]), static_cast<float>(last_action[1]),
    static_cast<float>(v / spec.v_scale), static_cast<float>(w / spec.w_scale)});
  if (spec.command_feedback) {
    out.insert(out.end(), {static_cast<float>(command[0] / spec.v_scale),
      static_cast<float>(command[1] / spec.w_scale)});
  }
  return out;
}

std::array<double, 2> commandFromAction(
  const DeploySpec & spec, const std::array<double, 2> & action, const std::array<double, 2> & cmd)
{
  const double a0 = std::clamp(action[0], -1.0, 1.0), a1 = std::clamp(action[1], -1.0, 1.0);
  const double lv = spec.a_max * spec.dt, lw = spec.alpha_max * spec.dt;
  if (spec.action_mode == "window") {
    return {std::clamp(cmd[0] + a0 * lv, spec.v_min, spec.v_max), std::clamp(cmd[1] + a1 * lw, -spec.w_max, spec.w_max)};
  }
  const double tv = spec.v_min + (a0 + 1.0) * 0.5 * (spec.v_max - spec.v_min), tw = a1 * spec.w_max;
  return {cmd[0] + std::clamp(tv - cmd[0], -lv, lv), cmd[1] + std::clamp(tw - cmd[1], -lw, lw)};
}

struct OnnxPolicy::Impl
{
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dwarl"};
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  DeploySpec spec;
};

OnnxPolicy::OnnxPolicy(const DeploySpec & spec, int threads)
: impl_(std::make_unique<Impl>())
{
  impl_->spec = spec;
  Ort::SessionOptions opt;
  opt.SetIntraOpNumThreads(threads);
  opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  impl_->session = std::make_unique<Ort::Session>(impl_->env, spec.onnx_file.c_str(), opt);
}

OnnxPolicy::~OnnxPolicy() = default;

std::array<double, 2> OnnxPolicy::operator()(const std::vector<float> & state, const std::vector<float> & lidar)
{
  const auto & s = impl_->spec;
  const int lidar_size = static_cast<int>(s.lidar_history.size()) * s.num_rays;
  if (static_cast<int>(state.size()) != s.obs_dim || static_cast<int>(lidar.size()) != lidar_size) {
    throw std::runtime_error("observation size mismatch");
  }
  const std::array<int64_t, 2> obs_shape{1, s.obs_dim};
  const std::array<int64_t, 4> lidar_shape{1, 1, static_cast<int64_t>(s.lidar_history.size()), s.num_rays};
  std::array<Ort::Value, 2> inputs{
    Ort::Value::CreateTensor<float>(impl_->mem, const_cast<float *>(state.data()), state.size(), obs_shape.data(),
      obs_shape.size()),
    Ort::Value::CreateTensor<float>(impl_->mem, const_cast<float *>(lidar.data()), lidar.size(),
      lidar_shape.data(), lidar_shape.size())};
  const char * in_names[] = {s.obs_input.c_str(), s.lidar_input.c_str()};
  const char * out_names[] = {s.action_output.c_str()};
  auto out = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 2, out_names, 1);
  const float * a = out[0].GetTensorData<float>();
  return {a[0], a[1]};
}

}  // namespace dwarl_controller
