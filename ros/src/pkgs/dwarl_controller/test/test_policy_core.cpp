// Replays golden.json from policy preparation (a noise-free episode recorded in the Python env) and checks
// that the C++ observation builder, ONNX policy and action mapping reproduce training bit-for-bit (float tol).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "dwarl_controller/policy_core.hpp"

using dwarl_controller::DeploySpec;
namespace fs = std::filesystem;

namespace
{
std::string exportDir()
{
  if (const char * d = std::getenv("DWARL_TEST_EXPORT_DIR")) {return d;}
  return "";
}
}  // namespace

TEST(Core, CommandRateLimit)
{
  DeploySpec s;
  auto c = dwarl_controller::commandFromAction(s, {1.0, 1.0}, {0.0, 0.0});
  EXPECT_NEAR(c[0], 0.1, 1e-12);  // a_max * dt
  EXPECT_NEAR(c[1], 0.3, 1e-12);
  c = dwarl_controller::commandFromAction(s, {0.0, 0.0}, {0.3, 0.0});
  EXPECT_NEAR(c[0], 0.3, 1e-12);  // a0 = 0 -> mid speed 0.3 m/s
}

TEST(Core, LidarRangeInvertsNormalization)
{
  DeploySpec s;
  dwarl_controller::Scan scan;
  scan.angle_min = -M_PI;
  scan.angle_increment = 2 * M_PI / s.num_rays;
  scan.range_max = 10.0;
  scan.ranges.assign(s.num_rays, 2.5f);
  for (const char * mode : {"linear", "log"}) {
    s.lidar_normalize = mode;
    s.angle_min = -M_PI;
    s.angle_increment = 2 * M_PI / s.num_rays;
    const auto o = dwarl_controller::lidarObservation(s, scan);
    EXPECT_NEAR(dwarl_controller::lidarRange(s, o[0]), 2.5, 1e-4) << mode;
  }
}

TEST(Core, StateObservationGoalPoints)
{
  DeploySpec s;
  s.lookahead_distances = {0.5, 1.0, 2.0};
  dwarl_controller::PathTracker tracker;
  tracker.setPlan({{0.0, 0.0}, {5.0, 0.0}});
  dwarl_controller::Vec2 goal, proj;
  std::vector<dwarl_controller::Vec2> goals;
  const auto obs = dwarl_controller::stateObservation(
    s, tracker, {1.0, 0.2, 0.0}, 0.0, 0.0, {0.0, 0.0}, {0.0, 0.0}, &goal, &goals, &proj);
  EXPECT_EQ(obs.size(), 13u);
  ASSERT_EQ(goals.size(), 3u);
  EXPECT_NEAR(goals[0].x, 1.5, 1e-9);
  EXPECT_NEAR(goals[2].x, 3.0, 1e-9);
  EXPECT_NEAR(proj.x, 1.0, 1e-9);
  EXPECT_NEAR(goal.x, 3.0, 1e-9);  // lookahead_distance 2.0
}

TEST(Core, CommandFeedbackAppendsNormalizedPreviousCommand)
{
  DeploySpec s;
  s.lookahead_distances = {2.0};
  s.command_feedback = true;
  dwarl_controller::PathTracker tracker;
  tracker.setPlan({{0.0, 0.0}, {5.0, 0.0}});
  const auto obs = dwarl_controller::stateObservation(
    s, tracker, {1.0, 0.0, 0.0}, 0.2, -0.3, {0.25, -0.5}, {0.4, -0.6});
  ASSERT_EQ(obs.size(), 9u);
  EXPECT_FLOAT_EQ(obs[3], 0.25f);  // original prefix remains action, measured velocity, then feedback
  EXPECT_FLOAT_EQ(obs[4], -0.5f);
  EXPECT_FLOAT_EQ(obs[5], 0.25f);
  EXPECT_FLOAT_EQ(obs[6], -0.2f);
  EXPECT_FLOAT_EQ(obs[7], 0.5f);
  EXPECT_FLOAT_EQ(obs[8], -0.4f);
  s.command_feedback = false;
  const auto legacy = dwarl_controller::stateObservation(
    s, tracker, {1.0, 0.0, 0.0}, 0.2, -0.3, {0.25, -0.5}, {0.4, -0.6});
  ASSERT_EQ(legacy.size(), 7u);
  EXPECT_TRUE(std::equal(legacy.begin(), legacy.end(), obs.begin()));
}

TEST(Core, DeployVersionContract)
{
  const auto dir = fs::path(::testing::TempDir()) / "dwarl_deploy_version_contract";
  fs::create_directories(dir);
  const auto write = [&](int version, bool feedback, bool tracker, int obs_dim) {
      std::ofstream f(dir / "deploy.yaml");
      f << "version: " << version << "\n"
        << "model:\n  onnx: policy.onnx\n  inputs:\n"
        << "  - {name: obs, shape: [1, " << obs_dim << "], group: state}\n"
        << "  - {name: lidar, shape: [1, 1, 1, 180], group: lidar}\n"
        << "  outputs: [{name: actions}]\n"
        << "control: {dt: 0.1}\n"
        << "lidar: {num_rays: 180, angle_min: -3.14, angle_increment: 0.01, min_range: 0.1, "
           "max_range: 10.0, origin: [0.0, 0.0], full_circle: true, history: [0], normalize: linear}\n"
        << "robot: {id: testbot, drive_type: diff}\n"
        << "state: {command_feedback: " << (feedback ? "true" : "false")
        << ", max_goal_distance: 20.0, v_scale: 0.8, w_scale: 1.5}\n"
        << "goal: {mode: lookahead, lookahead_distance: 2.0, lookahead_distances: [2.0], "
           "tolerance: 0.3, projection_window: [-5, 30]";
      if (tracker) {f << ", tracker: segment_window_v1";}
      f << "}\n"
        << "action: {mode: absolute, v: {min: -0.2, max: 0.8}, w: {max: 1.5}, "
           "rate_limit: {a_max: 1.0, alpha_max: 3.0}}\n";
    };
  write(5, true, true, 9);
  auto spec = DeploySpec::load(dir.string());
  EXPECT_TRUE(spec.command_feedback);
  EXPECT_EQ(spec.obs_dim, 9);
  write(3, false, false, 7);
  spec = DeploySpec::load(dir.string());
  EXPECT_FALSE(spec.command_feedback);
  EXPECT_EQ(spec.obs_dim, 7);
  write(4, true, true, 9);
  EXPECT_THROW(DeploySpec::load(dir.string()), std::runtime_error);
  write(5, true, false, 9);
  EXPECT_THROW(DeploySpec::load(dir.string()), std::runtime_error);
  write(3, true, true, 9);
  EXPECT_THROW(DeploySpec::load(dir.string()), std::runtime_error);
}

TEST(Core, LidarResampleShiftedLaser)
{
  DeploySpec s;
  s.num_rays = 4;
  s.angle_min = -M_PI;
  s.angle_increment = M_PI / 2;
  dwarl_controller::Scan scan;  // laser with 0 at the front, 4 rays, one wall 1 m ahead
  scan.angle_min = 0.0;
  scan.angle_increment = M_PI / 2;
  scan.range_max = 10.0;
  scan.ranges = {1.0f, std::numeric_limits<float>::infinity(), 2.0f, NAN};
  const auto o = dwarl_controller::lidarObservation(s, scan);
  const double span = s.max_range - s.min_range;
  EXPECT_NEAR(o[0], (2.0 - s.min_range) / span, 1e-6);  // behind
  EXPECT_NEAR(o[1], 1.0, 1e-6);  // right: nan -> max
  EXPECT_NEAR(o[2], (1.0 - s.min_range) / span, 1e-6);  // front
  EXPECT_NEAR(o[3], 1.0, 1e-6);  // left: inf -> max
}

TEST(Core, LidarBelowRangeMinIsNoReturn)
{
  DeploySpec s;
  s.num_rays = 4;
  s.angle_min = -M_PI;
  s.angle_increment = M_PI / 2;
  dwarl_controller::Scan scan;
  scan.angle_min = -M_PI;
  scan.angle_increment = M_PI / 2;
  scan.range_min = 0.12;
  scan.range_max = 10.0;
  scan.ranges = {0.0f, 0.05f, -std::numeric_limits<float>::infinity(), 1.0f};
  const auto o = dwarl_controller::lidarObservation(s, scan);
  EXPECT_NEAR(o[0], 1.0, 1e-6);  // 0.0 (invalid) -> no return
  EXPECT_NEAR(o[1], 1.0, 1e-6);  // below range_min -> no return
  EXPECT_NEAR(o[2], 0.0, 1e-6);  // -inf -> too close
}

TEST(Core, LidarMinPoolingKeepsThinObstacle)
{
  DeploySpec s;
  s.num_rays = 180;
  s.angle_min = -M_PI;
  s.angle_increment = 2 * M_PI / 180;
  dwarl_controller::Scan scan;  // 360 beams, one beam sees a pole at 1 m
  scan.angle_min = -M_PI;
  scan.angle_increment = 2 * M_PI / 360;
  scan.range_max = 10.0;
  scan.ranges.assign(360, std::numeric_limits<float>::infinity());
  for (int j : {181, 183}) {  // odd beams: dropped by nearest-beam resampling
    scan.ranges[j] = 1.0f;
    const auto o = dwarl_controller::lidarObservation(s, scan);
    const double span = s.max_range - s.min_range;
    double mn = 1.0;
    for (float v : o) {mn = std::min(mn, static_cast<double>(v));}
    EXPECT_NEAR(mn, (1.0 - s.min_range) / span, 1e-6);
    scan.ranges[j] = std::numeric_limits<float>::infinity();
  }
}

TEST(Core, LidarOffsetReprojected)
{
  DeploySpec s;
  s.num_rays = 4;
  s.angle_min = -M_PI;
  s.angle_increment = M_PI / 2;
  dwarl_controller::Scan scan;  // laser 0.1 m behind the centre sees a wall 1.1 m ahead of it
  scan.angle_min = -M_PI;
  scan.angle_increment = M_PI / 2;
  scan.range_max = 10.0;
  const float I = std::numeric_limits<float>::infinity();
  scan.ranges = {I, I, 1.1f, I};
  const auto o = dwarl_controller::lidarObservation(s, scan, dwarl_controller::LaserPose::planar(-0.1, 0.0, 0.0));
  EXPECT_NEAR(o[2], (1.0 - s.min_range) / (s.max_range - s.min_range), 1e-6);  // 1.0 m from the centre
}

TEST(Core, LidarUpsideDownAtRayOrigin)
{
  DeploySpec s;  // training rays start 0.4 m ahead of the base origin
  s.num_rays = 4;
  s.angle_min = -M_PI;
  s.angle_increment = M_PI / 2;
  s.origin_x = 0.4;
  dwarl_controller::Scan scan;  // upside-down laser at the ray origin: its +90 deg beam points to the robot's right
  scan.angle_min = -M_PI;
  scan.angle_increment = M_PI / 2;
  scan.range_max = 10.0;
  const float I = std::numeric_limits<float>::infinity();
  scan.ranges = {I, I, I, 2.0f};  // laser +90 deg
  const auto laser = dwarl_controller::LaserPose::fromQuaternion(0.4, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0);  // roll pi
  const auto o = dwarl_controller::lidarObservation(s, scan, laser);
  EXPECT_NEAR(o[1], (2.0 - s.min_range) / (s.max_range - s.min_range), 1e-6);  // robot right (-90 deg)
  EXPECT_NEAR(o[3], 1.0, 1e-6);  // robot left: nothing
}

TEST(Golden, ReproducesTraining)
{
  const auto dir = exportDir();
  if (dir.empty() || !fs::exists(fs::path(dir) / "golden.json")) {
    GTEST_SKIP() << "no policy bundle with golden.json (run `dwarl test` to prepare a checkpoint)";
  }
  const auto spec = DeploySpec::load(dir);
  std::ifstream f(fs::path(dir) / "golden.json");
  const auto g = nlohmann::json::parse(f);
  std::vector<dwarl_controller::Vec2> plan;
  for (const auto & p : g["plan"]) {plan.push_back({p[0], p[1]});}
  dwarl_controller::PathTracker tracker;
  tracker.setPlan(plan);
  dwarl_controller::OnnxPolicy policy(spec);

  const size_t n = g["action"].size();
  ASSERT_GT(n, 5u);
  std::vector<std::vector<float>> scan_buf;
  for (size_t t = 0; t < n; ++t) {
    SCOPED_TRACE("step " + std::to_string(t));
    dwarl_controller::Scan scan;  // the sim scan is already in the training layout, measured from the ray origin
    scan.angle_min = spec.angle_min;
    scan.angle_increment = spec.angle_increment;
    scan.range_max = spec.max_range;
    for (double r : g["ranges"][t]) {scan.ranges.push_back(static_cast<float>(r));}
    const auto current_lidar = dwarl_controller::lidarObservation(
      spec, scan, dwarl_controller::LaserPose::planar(spec.origin_x, spec.origin_y, 0.0));
    const size_t depth = static_cast<size_t>(spec.lidar_history.back() + 1);
    if (scan_buf.empty()) {
      scan_buf.assign(depth, current_lidar);
    } else {
      scan_buf.insert(scan_buf.begin(), current_lidar);
      scan_buf.resize(depth);
    }
    std::vector<float> lidar;
    for (const int lag : spec.lidar_history) {
      lidar.insert(lidar.end(), scan_buf[lag].begin(), scan_buf[lag].end());
    }
    for (size_t i = 0; i < lidar.size(); ++i) {
      ASSERT_NEAR(lidar[i], g["obs_lidar"][t][i].get<double>(), 1e-5) << "lidar " << i;
    }
    const auto & ps = g["pose"][t];
    const auto & la = g["last_action"][t];
    const auto & cmd = g["cmd"][t];
    const auto state = dwarl_controller::stateObservation(
      spec, tracker, {ps[0], ps[1], ps[2]}, g["vel"][t][0], g["vel"][t][1], {la[0], la[1]},
      {cmd[0], cmd[1]});
    for (int i = 0; i < spec.obs_dim; ++i) {
      ASSERT_NEAR(state[i], g["obs_state"][t][i].get<double>(), 1e-4) << "state " << i;
    }
    // policy on the recorded observation (isolates the network from observation rounding)
    std::vector<float> rs, rl;
    for (double v : g["obs_state"][t]) {rs.push_back(static_cast<float>(v));}
    for (double v : g["obs_lidar"][t]) {rl.push_back(static_cast<float>(v));}
    const auto a = policy(rs, rl);
    ASSERT_NEAR(a[0], g["action"][t][0].get<double>(), 1e-4);
    ASSERT_NEAR(a[1], g["action"][t][1].get<double>(), 1e-4);
    const auto c = dwarl_controller::commandFromAction(
      spec, {g["action"][t][0], g["action"][t][1]}, {g["cmd"][t][0], g["cmd"][t][1]});
    ASSERT_NEAR(c[0], g["cmd_next"][t][0].get<double>(), 1e-9);
    ASSERT_NEAR(c[1], g["cmd_next"][t][1].get<double>(), 1e-9);
  }
}
