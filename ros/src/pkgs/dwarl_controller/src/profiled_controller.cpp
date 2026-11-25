// Optional common timing boundary for all benchmark controllers.
#include <chrono>
#include "nav2_core/controller.hpp"
#include "pluginlib/class_loader.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"

namespace dwarl_controller
{
class ProfiledController : public nav2_core::Controller
{
public:
  void configure(const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent, std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap) override
  {
    auto node = parent.lock();
    if (!node->has_parameter(name + ".wrapped_plugin")) {
      node->declare_parameter(name + ".wrapped_plugin", "dwarl_controller::DwarlController");
    }
    const auto type = node->get_parameter(name + ".wrapped_plugin").as_string();
    if (type == "dwarl_controller::ProfiledController") {
      throw std::runtime_error("ProfiledController cannot wrap itself");
    }
    loader_ = std::make_unique<pluginlib::ClassLoader<nav2_core::Controller>>("nav2_core", "nav2_core::Controller");
    inner_ = loader_->createSharedInstance(type);
    inner_->configure(parent, name, tf, costmap);
    clock_ = node->get_clock();
    publisher_ = node->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(name + "/compute_metrics", 100);
  }
  void activate() override {inner_->activate(); publisher_->on_activate();}
  void deactivate() override {inner_->deactivate(); publisher_->on_deactivate();}
  void cleanup() override {inner_->cleanup(); publisher_.reset(); inner_.reset(); loader_.reset();}
  void reset() override {inner_->reset();}
  void setPlan(const nav_msgs::msg::Path & path) override {inner_->setPlan(path);}
  void setSpeedLimit(const double & limit, const bool & percentage) override
  {inner_->setSpeedLimit(limit, percentage);}
  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * checker) override
  {
    const auto start = std::chrono::steady_clock::now();
    try {
      auto result = inner_->computeVelocityCommands(pose, velocity, checker);
      publish(start, false);
      return result;
    } catch (...) {
      publish(start, true);
      throw;
    }
  }
private:
  void publish(std::chrono::steady_clock::time_point start, bool failed)
  {
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    diagnostic_msgs::msg::DiagnosticArray message;
    message.header.stamp = clock_->now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "controller_compute";
    diagnostic_msgs::msg::KeyValue value;
    value.key = "controller_ms"; value.value = std::to_string(ms); status.values.push_back(value);
    value.key = "failed"; value.value = failed ? "1" : "0"; status.values.push_back(value);
    message.status.push_back(status);
    publisher_->publish(message);
  }
  // Loader must outlive its plugin instance.
  std::unique_ptr<pluginlib::ClassLoader<nav2_core::Controller>> loader_;
  std::shared_ptr<nav2_core::Controller> inner_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp_lifecycle::LifecyclePublisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr publisher_;
};
}  // namespace dwarl_controller
PLUGINLIB_EXPORT_CLASS(dwarl_controller::ProfiledController, nav2_core::Controller)
