from abc import abstractmethod
from typing import Dict, Protocol

import numpy as np
import time

class Robot(Protocol):
    """Robot protocol.

    A protocol for a robot that can be controlled.
    """

    @abstractmethod
    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        raise NotImplementedError

    @abstractmethod
    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        raise NotImplementedError

    @abstractmethod
    def command_joint_state(self, joint_state: np.ndarray, flag_in: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            flag_in:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        raise NotImplementedError

    @abstractmethod
    def command_movel(self, pose: np.ndarray, flag_in: np.ndarray) -> None:
        """Command robot TCP to a Cartesian pose.

        Args:
            pose (np.ndarray): pose [x, y, z, rx, ry, rz], mm + deg.
            flag_in (np.ndarray): [left_enable, right_enable] for bimanual robot.
        """
        raise NotImplementedError

    @abstractmethod
    def command_gripper(self, position, flag_in: np.ndarray = None) -> None:
        """Command robot gripper opening ratio.

        Args:
            position: target opening ratio in [0, 1].
            flag_in (np.ndarray): [left_enable, right_enable] for bimanual robot.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the robot.

        This is to extract all the information that is available from the robot,
        such as joint positions, joint velocities, etc. This may also include
        information from additional sensors, such as cameras, force sensors, etc.

        Returns:
            Dict[str, np.ndarray]: A dictionary of observations.
        """
        raise NotImplementedError

    def set_do_status(self, which_do, arm=None):
        """Get the current observations of the robot.

        Args:
            which_do:
            arm:
        """
        raise NotImplementedError

    def get_DI_state(self, i):
        raise NotImplementedError

    @abstractmethod
    def get_XYZrxryrz_state(self) -> np.ndarray:
        """Get the current X Y Z rx ry rz state of the robot.
        Returns:
            T: The current X Y Z rx ry rz state of the leader robot.
        """
        raise NotImplementedError


class PrintRobot(Robot):
    """A robot that prints the commanded joint state."""

    def __init__(self, num_dofs: int, dont_print: bool = False):
        self._num_dofs = num_dofs
        # self._joint_state = np.zeros(num_dofs)
        self._joint_state = np.deg2rad(
            [90, 0, 90, 0, -90, 0, 90]
        )  # DOBOT

        self._dont_print = dont_print

    def num_dofs(self) -> int:
        return self._num_dofs

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray, flag_in: np.ndarray = None) -> None:
        assert len(joint_state) == (self._num_dofs), (
            f"Expected joint state of length {self._num_dofs}, "
            f"got {len(joint_state)}."
        )
        self._joint_state = joint_state
        if not self._dont_print:
            print(self._joint_state)

    def command_movel(self, pose: np.ndarray, flag_in: np.ndarray = None) -> None:
        pose = np.asarray(pose, dtype=float).flatten()
        assert len(pose) >= 6, f"Expected pose length >= 6, got {len(pose)}"
        if not self._dont_print:
            print({"movel_pose": [float(v) for v in pose[:6]], "flag_in": flag_in})
        return 1

    def command_gripper(self, position, flag_in: np.ndarray = None) -> None:
        values = np.asarray(position, dtype=float).flatten()
        target = float(values[0]) if len(values) > 0 else 1.0
        target = float(np.clip(target, 0.0, 1.0))
        if not self._dont_print:
            print({"gripper_position": target, "flag_in": flag_in})
        return 1

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_state = self.get_joint_state()
        pos_quat = np.zeros(7)
        return {
            "joint_positions": joint_state,
            "joint_velocities": joint_state,
            "ee_pos_quat": pos_quat,
            "gripper_position": np.array(0),
        }


class BimanualRobot(Robot):
    def __init__(self, robot_l: Robot, robot_r: Robot):
        self._robot_l = robot_l
        self._robot_r = robot_r

        self.frequency_ = 1/0.015
        # Set delta time to be used by receiveCallback
        self.delta_time_ = 1 / self.frequency_
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"

    def num_dofs(self) -> int:
        return self._robot_l.num_dofs() + self._robot_r.num_dofs()

    def get_joint_state(self) -> np.ndarray:
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"
        return np.concatenate(
            (self._robot_l.get_joint_state(), self._robot_r.get_joint_state())
        )

    @staticmethod
    def _normalize_flag_in(flag_in):
        if flag_in is None:
            return [1, 0]
        arr = np.asarray(flag_in, dtype=int).flatten().tolist()
        if len(arr) < 2:
            arr = arr + [0] * (2 - len(arr))
        return [1 if arr[0] else 0, 1 if arr[1] else 0]

    def command_joint_state(self, joint_state: np.ndarray, flag_in) -> None:
        t_start = time.time()
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"
        normalized_flag = self._normalize_flag_in(flag_in)
        joints = np.asarray(joint_state, dtype=float).flatten()
        left_n = int(self._robot_l.num_dofs())
        right_n = int(self._robot_r.num_dofs())
        if normalized_flag[0]:
            if len(joints) >= left_n + right_n:
                left_joint_state = joints[:left_n]
            elif len(joints) >= left_n:
                left_joint_state = joints[:left_n]
            else:
                raise ValueError(f"left joint_state length invalid: got {len(joints)}, need >= {left_n}")
            self._robot_l.command_joint_state(left_joint_state)
        if normalized_flag[1]:
            if len(joints) >= left_n + right_n:
                right_joint_state = joints[left_n : left_n + right_n]
            elif len(joints) >= right_n and not normalized_flag[0]:
                right_joint_state = joints[:right_n]
            else:
                raise ValueError(
                    f"right joint_state length invalid: got {len(joints)}, "
                    f"need >= {left_n + right_n} (or >= {right_n} when only right enabled)"
                )
            self._robot_r.command_joint_state(right_joint_state)
        # t_start2 = time.time()
        return 1

    def command_movel(self, pose: np.ndarray, flag_in) -> None:
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"
        normalized_flag = self._normalize_flag_in(flag_in)
        pose_vec = np.asarray(pose, dtype=float).flatten()
        if len(pose_vec) < 6:
            raise ValueError(f"pose length invalid: got {len(pose_vec)}, need >= 6")
        if normalized_flag[0] and normalized_flag[1]:
            # Allow [Lx..Lrz,Rx..Rrz] in one payload, otherwise broadcast the first pose.
            left_pose = pose_vec[:6]
            right_pose = pose_vec[6:12] if len(pose_vec) >= 12 else pose_vec[:6]
        elif normalized_flag[0]:
            left_pose = pose_vec[:6]
            right_pose = None
        elif normalized_flag[1]:
            left_pose = None
            right_pose = pose_vec[:6]
        else:
            return 1
        if left_pose is not None:
            self._robot_l.command_movel(left_pose)
        if right_pose is not None:
            self._robot_r.command_movel(right_pose)
        return 1

    def command_gripper(self, position, flag_in=None) -> None:
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"
        normalized_flag = self._normalize_flag_in(flag_in)
        position_vec = np.asarray(position, dtype=float).flatten()

        def _target(idx: int) -> float:
            if len(position_vec) <= 0:
                value = 1.0
            elif len(position_vec) == 1:
                value = float(position_vec[0])
            else:
                value = float(position_vec[idx])
            return float(np.clip(value, 0.0, 1.0))

        if normalized_flag[0]:
            self._robot_l.command_gripper(_target(0))
        if normalized_flag[1]:
            right_idx = 1 if len(position_vec) > 1 else 0
            self._robot_r.command_gripper(_target(right_idx))
        return 1

    def get_observations(self) -> Dict[str, np.ndarray]:
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"
        l_obs = self._robot_l.get_observations()
        r_obs = self._robot_r.get_observations()
        assert l_obs.keys() == r_obs.keys()
        return_obs = {}
        for k in l_obs.keys():
            try:
                return_obs[k] = np.concatenate((l_obs[k], r_obs[k]))
            except Exception as e:
                print(e)
                print(k)
                print(l_obs[k])
                print(r_obs[k])
                raise RuntimeError()

        return return_obs

    def set_do_status(self, which_do, arm=None):
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"
        arm_name = str(arm or "left").strip().lower()
        if arm_name in {"both", "bimanual", "lr"}:
            self._robot_l.set_do_status(which_do)
            self._robot_r.set_do_status(which_do)
            return 1
        if arm_name in {"right", "r"}:
            self._robot_r.set_do_status(which_do)
            return 1
        # Keep historical behavior as default: left arm only.
        self._robot_l.set_do_status(which_do)
        return 1

    def get_XYZrxryrz_state(self) -> np.ndarray:
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"
        return np.concatenate(
            (self._robot_l.get_XYZrxryrz_state(), self._robot_r.get_XYZrxryrz_state())
        )

def main():
    pass


if __name__ == "__main__":
    main()
