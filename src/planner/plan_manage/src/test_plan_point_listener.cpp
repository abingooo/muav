#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

#include <queue>
#include <sstream>

class PlanPointResponder
{
public:
  PlanPointResponder()
    : nh_private_("~"),
      ack_delay_(nh_private_.param("ack_delay", 5.0)),
      received_count_(0),
      timer_active_(false)
  {
    point_topic_ = nh_private_.param<std::string>("topic", "/toplan/single_plan_point");
    feedback_topic_ = nh_private_.param<std::string>("feedback_topic", "/toplan/waypoint_done");

    point_sub_ = nh_.subscribe(point_topic_, 10, &PlanPointResponder::pointCallback, this);
    feedback_pub_ = nh_.advertise<std_msgs::String>(feedback_topic_, 10);

    ROS_INFO("PlanPointResponder listening on %s and publishing feedback to %s",
             point_topic_.c_str(), feedback_topic_.c_str());
  }

private:
  void pointCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
  {
    received_count_++;
    pending_ids_.push(received_count_);

    const auto &p = msg->pose.position;
    ROS_INFO("Received plan point #%d: (%.2f, %.2f, %.2f) frame=%s",
             received_count_, p.x, p.y, p.z, msg->header.frame_id.c_str());

    if (!timer_active_)
    {
      startTimer();
    }
  }

  void timerCallback(const ros::TimerEvent &)
  {
    if (pending_ids_.empty())
    {
      timer_active_ = false;
      return;
    }

    const int id = pending_ids_.front();
    pending_ids_.pop();

    std_msgs::String msg;
    std::stringstream ss;
    ss << "arrived_" << id;
    msg.data = ss.str();
    feedback_pub_.publish(msg);
    ROS_INFO("Published waypoint_done: %s", msg.data.c_str());

    timer_active_ = false;
    if (!pending_ids_.empty())
    {
      startTimer();
    }
  }

  void startTimer()
  {
    timer_ = nh_.createTimer(ros::Duration(ack_delay_), &PlanPointResponder::timerCallback, this, true);
    timer_active_ = true;
  }

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  ros::Subscriber point_sub_;
  ros::Publisher feedback_pub_;
  ros::Timer timer_;
  std::queue<int> pending_ids_;
  std::string point_topic_;
  std::string feedback_topic_;
  double ack_delay_;
  int received_count_;
  bool timer_active_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_plan_point_responder_cpp");
  PlanPointResponder responder;
  ros::spin();
  return 0;
}
