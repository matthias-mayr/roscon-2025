#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from rclpy.action import ActionClient

from niryo_ned_ros2_interfaces.srv import ControlConveyor
from niryo_ned_ros2_interfaces.msg import ArmMoveCommand, DigitalIOState, ToolCommand
from niryo_ned_ros2_interfaces.action import RobotMove, Tool

import yaml
import os
from ament_index_python.packages import get_package_share_directory

class QualityCheckNode(Node):
    def __init__(self):
        super().__init__("quality_check_node")

        self._object_detected : bool = False

        # --- Parameters ---
        self.conveyor_id = self.declare_parameter("conveyor_id", 9).get_parameter_value().integer_value
        self.speed = self.declare_parameter("speed", 60).get_parameter_value().integer_value
        self.sensor_index = self.declare_parameter("sensor_index", 4).get_parameter_value().integer_value
        self.digital_state_topic = self.declare_parameter(
            "digital_state_topic", "/niryo_robot_rpi/digital_io_state"
        ).get_parameter_value().string_value
        self.conveyor_service = self.declare_parameter(
            "conveyor_service", "/niryo_robot/conveyor/control_conveyor"
        ).get_parameter_value().string_value
        self.robot_action = self.declare_parameter(
            "robot_action", "/niryo_robot_arm_commander/robot_action"
        ).get_parameter_value().string_value
        self.tool_action = self.declare_parameter(
            "tool_action", "/niryo_robot_tools_commander/action_server"
        ).get_parameter_value().string_value
        self.tool_id = self.declare_parameter("tool_id", 11).get_parameter_value().integer_value
        self.max_torque_percentage = self.declare_parameter("max_torque_percentage", 100).get_parameter_value().integer_value
        self.hold_torque_percentage = self.declare_parameter("hold_torque_percentage", 100).get_parameter_value().integer_value

        # --- Poses ---
        default_poses_path = os.path.join(
            get_package_share_directory("workshop_quality_check_manager"), "config", "poses.yaml"
        )
        poses_path = self.declare_parameter("poses_path", default_poses_path).get_parameter_value().string_value
        with open(poses_path, "r") as f:
            poses_file = yaml.safe_load(f)
        poses = poses_file.get("poses", {})

        # --- Helpers ---
        self.conveyor = ConveyorController(self, self.conveyor_service, self.conveyor_id, self.speed)
        tool_cfg = {"id": self.tool_id, "max": self.max_torque_percentage, "hold": self.hold_torque_percentage}
        self.pick_place = PickAndPlaceExecutor(self, self.robot_action, self.tool_action, poses, tool_cfg)

        # --- State ---
        self._last_object_detected = None
        self._last_safety_state: str | None = None

        # --- Subscription ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(DigitalIOState, self.digital_state_topic, self._on_digital_state, qos)
        self.create_subscription(String, "/safety_state", self._on_safety_state, 10)

        self.get_logger().info("quality_check_node_sync started")

    def _on_digital_state(self, msg: DigitalIOState) -> None:
        self._object_detected = not msg.digital_inputs[-1].value

    def _on_safety_state(self, msg: String) -> None:
        # TODO: Implement the safety state method
        pass

    def run_loop(self):
        self.conveyor.set_running(True)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            self.conveyor.set_running(not self._object_detected)


class PickAndPlaceExecutor:

    def __init__(self, node: Node, robot_action: str, tool_action: str, poses: dict, tool_cfg: dict) -> None:
        self._node = node
        self._robot = ActionClient(node, RobotMove, robot_action)
        self._tool = ActionClient(node, Tool, tool_action)
        self._poses = poses
        self._tool_cfg = tool_cfg

        if not self._robot.wait_for_server(timeout_sec=5.0):
            self._node.get_logger().error(f"robot server {robot_action} not available !")
        if not self._tool.wait_for_server(timeout_sec=5.0):
            self._node.get_logger().error(f"tool server {tool_action} not available !")


    def _move(self, joints):
        goal = RobotMove.Goal()
        goal_cmd = ArmMoveCommand()
        goal_cmd.joints = joints
        goal_cmd.cmd_type = ArmMoveCommand.JOINTS
        goal.cmd = goal_cmd
        return self._send_goal_async(self._robot, goal)

    def _tool_cmd(self, activate: bool):
        tool_goal = Tool.Goal()
        tool_cmd = ToolCommand()
        tool_cmd.tool_id = self._tool_cfg["id"]
        tool_cmd.max_torque_percentage = self._tool_cfg["max"]
        tool_cmd.hold_torque_percentage = self._tool_cfg["hold"]
        tool_cmd.cmd_type = ToolCommand.OPEN_GRIPPER if activate else ToolCommand.CLOSE_GRIPPER
        tool_goal.cmd = tool_cmd
        return self._send_goal_async(self._tool, tool_goal)

    def _send_goal_async(self, action_client: ActionClient, goal) -> None:
        send_future = action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self._node.get_logger().error("command rejected")
            return
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future)
        return result_future.result().result


class ConveyorController:

    def __init__(self, node: Node, service_name: str, conveyor_id: int, speed: int) -> None:
        self._node = node
        self._client = node.create_client(ControlConveyor, service_name)
        self._conveyor_id = conveyor_id
        self._speed = speed
        self._current_state = None
        self.last_command = None

        if not self._client.wait_for_service(timeout_sec=5.0):
            self._node.get_logger().error(f"Service {service_name} not available !")

    def set_running(self, run: bool) -> None:
        # print(f'Conveyor running {run}')
        if self.last_command != run:
            self._node.get_logger().info(f'Setting conveyor running: {run} from {self.last_command}')
            self.last_command = run
        req = ControlConveyor.Request()
        req.id = self._conveyor_id
        req.control_on = True
        req.speed = self._speed
        req.direction = (1 if run else 0)
        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)



def main():
    rclpy.init()
    node = QualityCheckNode()
    try:
        node.run_loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

