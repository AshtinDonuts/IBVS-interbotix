#!/usr/bin/env python3

# Copyright 2024 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys

from rclpy import node

from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

"""
This script makes the end-effector draw a square in 3D space.
Note that this script may not work for every arm as it was designed for the wx250.
Make sure to adjust commanded joint positions and poses as necessary.

To get started, open a terminal and type:

    ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=aloha_vx300s use_sim:=true

Then change to this directory and type:

    python3 simple_forward.py
    
"""

def main():

    bot = InterbotixManipulatorXS(
        robot_model='aloha_vx300s',
        # robot_name='follower_solo',
        group_name='arm',
        gripper_name='gripper',
        require_gravity_torques=False,
    )
    # DEBUG
    # Set default moving time to slow down initial motions
    bot.arm.moving_time = 8.0
    bot.arm.accel_time = 4.0

    robot_startup()

    # =======  USEFUL POSES ====== #

    # normal Teleop Start pose
    # joint_positions = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
    
    # Feb 6 - Puzzle piece servo start pose
    joint_positions = [0.046, -0.60, 0.85, 0.003, 0.0, 0.006]

    # Feb 6 - Lean back pose (Leader Safe!!)
    # joint_positions = [0.0, -1.322, 0.655, -0.018, 0.920, -0.061]

    # SLEEP POSE  ( alternatively, use go_sleep.py )
    sleep_joint_positions = [-0.012, -1.850, 1.529, -0.054, -1.839, 0.064]

    # ============================= #

    bot.arm.set_joint_positions(joint_positions, moving_time=4.0)
    # bot.arm.set_joint_positions(sleep_joint_positions, moving_time=4.0)

    # bot.arm.go_to_sleep_pose()

    robot_shutdown()


if __name__ == '__main__':
    main()
