sudo chmod 777 /dev/ttyACM0 & sleep 1;
# roslaunch realsense2_camera rs_camera.launch & sleep 10;
roslaunch realsense2_camera rs_camera.launch align_depth:=true & sleep 2;
roslaunch mavros px4.launch & sleep 3;
roslaunch vins fast_drone_250.launch
wait;
