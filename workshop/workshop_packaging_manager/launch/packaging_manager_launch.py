import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

from rclpy import logging

logger = logging.get_logger("packaging_manager.launch")

PACKAGE_NAME = "workshop_packaging_manager"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

def generate_launch_description():
    # Arguments:
    conveyor_id_arg = DeclareLaunchArgument(
        "conveyor_id",
        default_value="9",
        description="ID of the conveyor to use",
    )
    speed_arg = DeclareLaunchArgument(
        "speed",
        default_value="60",
        description="Speed of the conveyor in percentage of the maximum velocity.",
    )
    sensor_index_arg = DeclareLaunchArgument(
        "sensor_index",
        default_value="4",
        description="Index of the digital input where the sensor is connected.",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value="moveit.rviz",
        description="RViz configuration file",
    )

    
    urdf_file = os.path.join(
        get_package_share_directory("niryo_ned_description"),
        "urdf/ned3pro",
        "niryo_ned3pro.urdf.xacro",
    )

    # Build MoveIt2 configuration
    moveit_config = (
        MoveItConfigsBuilder("niryo_ned3pro", package_name=PACKAGE_NAME)
        .robot_description(file_path=urdf_file)
        .joint_limits(file_path=os.path.join(PACKAGE_PATH,
                "config", "joint_limits.yaml"))
        .robot_description_semantic(file_path=os.path.join(PACKAGE_PATH,
                "config", "niryo_ned3pro.srdf"))
        .robot_description_kinematics(file_path=os.path.join(PACKAGE_PATH,
                "config", "kinematics.yaml"))
        .trajectory_execution(file_path=os.path.join(PACKAGE_PATH,
                "config", "moveit_controllers.yaml"))
        .moveit_cpp(file_path=os.path.join(PACKAGE_PATH,
                "config", "moveit_py_config.yaml"))
        .to_moveit_configs()
    )

    packaging_node = Node(
        name="packaging_node",
        package=PACKAGE_NAME,
        executable="packaging_node.py",
        output="both",
        parameters=[
            moveit_config.to_dict(),
            {
                "conveyor_id": LaunchConfiguration("conveyor_id"),
                "speed": LaunchConfiguration("speed"),
                "sensor_index": LaunchConfiguration("sensor_index"),
                "digital_state_topic": "/niryo_robot_rpi/digital_io_state",
                "conveyor_service": "/niryo_robot/conveyor/control_conveyor",
            }
        ],
    )

    # RViz configuration
    rviz_base = LaunchConfiguration("rviz_config")
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("workshop_packaging_manager"), "config", rviz_base]
    )
    # RViz node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
        ],
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["--frame-id", "world", "--child-frame-id", "base_link"],
    )

    return LaunchDescription([
        rviz_config_arg,
        conveyor_id_arg,
        speed_arg,
        sensor_index_arg,
        static_tf,
        # rviz_node,
        packaging_node,
    ])