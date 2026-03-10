import mujoco
from mujoco.glfw import glfw

import numpy as np
from typing import List
from scipy.spatial.transform import Rotation


class MuJoCoVisualizer():
    """ This class provides a visualizer for the MuJoCo Libary. It takes in a MuJoCo Scene at instantiation and will launch a visualizer window. 
    Use the ``simulate()`` to run the simulation.
    """

    def __init__(self, xml_path: str):
        """ Consturctor for the MuJoCoVisualizer class. Sets up window and initializes MuJoCo sim data structures.

        :param xml_path: Path to the xml file that describes the MuJoCo scene.
        :type xml_path: str

        :returns: MuJoCoVisualizer object
        :rtype: MuJoCoVisualizer
        """

        # Get the file name from the xml path -- used to set window name
        if '/' in xml_path:
            # if xml path contains a slash, set model name to be everything after the last slash and before the .xml
            self.model_name = xml_path.split('/')[-1].split('.')[0]
        else:
            # otherwise, just set it to everything before the .xml
            self.model_name = xml_path.split('.')[0]

        self.framerate = 60.0

        # Initialize MuJoCo data structures
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Load initial keyframe (if present) so the arm starts in the right config
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            # Fallback: set Sawyer arm joints to a safe starting config
            self.data.qpos[:7] = [0.0, -1.1775, 0.0, 2.1761, 0.0, 0.5663, 3.3124]
        mujoco.mj_forward(self.model, self.data)

        # Init Visualization data structures
        self.camera = mujoco.MjvCamera()           # Abstract camera
        self.viz_options = mujoco.MjvOption()      # visualization options

        # Initialize GLFW (the viz window)
        glfw.init()
        self.window = glfw.create_window(1200, 900, self.model_name, None, None)  # TODO: size params here??
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        mujoco.mjv_defaultCamera(self.camera)
        # Set custom camera view
        self.camera.azimuth = 180  # left-right rotation around the model
        self.camera.elevation = -30  # up-down rotation
        self.camera.distance = 2.5  # zoom out (larger = further away)
        self.camera.lookat = np.array([0.0, 0.0, 0.8])  # where the camera is looking (world coordinates)
        
        mujoco.mjv_defaultOption(self.viz_options)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # Add GLFW mouse and keyboard callbacks
        # glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move)
        glfw.set_mouse_button_callback(self.window, self._mouse_button)
        glfw.set_scroll_callback(self.window, self._scroll)

        # For callback functions
        self.button_left: bool = False
        self.button_middle: bool = False
        self.button_right: bool = False
        self.lastx: float = 0
        self.lasty: float = 0

        # for PID control
        self.Kp: float = 5.0
        self.Ki: float = 2.0
        self.Kd: float = 1.0

        self.num_joints = 9

        self.integral: np.array = np.zeros(self.model.nq)[:self.num_joints] # size of number of joints
        self.prev_error: np.array = np.zeros(self.model.nq)[:self.num_joints]

        # Current joint target that the PID controller drives toward.
        # Initialised to the keyframe / starting qpos so the arm holds position.
        self._target: np.ndarray = self.data.qpos[:self.num_joints].copy()

    def add_target_to_trajectory(self, target: List[float]) -> None:
        """Set a new joint-position target for the PID controller.

        :param target: 7 arm joint positions [right_j0 .. right_j6].
        """
        if len(target) != 7:
            raise ValueError("Target should be an array of length 7 (joints).")
        target_complete = np.concatenate((target, np.zeros(2)))  # append gripper joints
        # Reset PID state when target changes significantly to avoid integral windup
        if np.linalg.norm(target_complete - self._target) > 0.01:
            self.integral = np.zeros(self.num_joints)
            self.prev_error = np.zeros(self.num_joints)
        self._target = target_complete.copy()

    def operate_gripper(self, open: bool = True) -> None:
        """ Open or close the gripper.

        :param open: If True, open the gripper; if False, close the gripper.
        :type open: bool
        """
        # Indices 7 and 8 are typically the two gripper joints for Sawyer
        left_finger_idx = 7
        right_finger_idx = 8

        if open:
            # Positive values open the gripper (you might need to tune exact values depending on your model)
            target_gripper = np.array([0.04, 0.04])  # ~4 cm open
        else:
            # Close the gripper
            target_gripper = np.array([0.0, 0.0])    # 0 cm, fingers closed

        # Set the desired qpos for the gripper joints
        self._target[left_finger_idx] = target_gripper[0]
        self._target[right_finger_idx] = target_gripper[1]

    def get_pose(self) -> List[float]:
        """ Returns the current pose of the robot.

        :returns: Current position of the robot.
        :rtype: List of x, y, z position

        :returns: Current orientation of the robot.
        :rtype: list of x, y, z, w quaternion
        """
        site_name = "end_effector"
        site_id = self.model.site(site_name).id

        position = self.data.site_xpos[site_id]
        rotation_matrix = self.data.site_xmat[site_id].reshape(3,3)

        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()

        return position, quaternion


    def get_object_position(self, object_name: str) -> np.ndarray:
        """ Returns the position of a specific object in the scene.

        :param object_name: Name of the object in the MuJoCo XML file.
        :type object_name: str
        :returns: Position of the object as an [x, y, z] numpy array.
        :rtype: np.ndarray
        """
        try:
            # Get the body ID from the model using the object name
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
            # Retrieve the position of the body using its ID
            position = self.data.xpos[body_id]
            return position
        except mujoco.FatalError:
            print(f"Object '{object_name}' not found in the MuJoCo model.")
            return np.array([np.nan, np.nan, np.nan])

    def set_trajectory(self, trajectory: List[List[float]]) -> None:
        """Set the last waypoint of a trajectory as the current target."""
        if trajectory:
            last = np.array(trajectory[-1], dtype=np.float64)
            self._target = last[:self.num_joints].copy()


    def _scroll(self, window, xoffset: float, yoffset: float) -> None:
        """ Callback function for mouse scroll wheel. Zooms in and out of the scene.

        :param window: GLFW window object that initiates the callback
        :type window: GLFW window object

        :param xoffset: x offset of the mouse scroll wheel
        :type xoffset: float

        :param yoffset: y offset of the mouse scroll wheel
        :type yoffset: float
        """
        action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.model, action, 0.0, 0.05 * yoffset, self.scene, self.camera)


    def _mouse_button(self, window, button: int, act: int, mods:int) -> None:
        """ Callback function for mouse button clicks. Updates the state of the mouse buttons.

        :param window: GLFW window object that initiates the callback
        :type window: GLFW window object

        :param button: button that was clicked
        :type button: int (enum for mouse buttons)

        :param act: action that was performed on the button (press or release)
        :type act: int (enum for mouse actions)

        :param mods: modifier keys that were pressed (shift, ctrl, etc.)
        :type mods: int (enum for modifier keys)
        """

        # update button state
        self.button_left   = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)   == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right  = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT)  == glfw.PRESS)


    def _mouse_move(self, window, xpos: float, ypos: float) -> None:
        """ Callback function for mouse movement. Moves the camera around the scene when the mouse is clicked and dragged.
            Left button for move, right button for rotate.
        
        :param window: GLFW window object the initiated the callback
        :type window: GLFW window object

        :param xpos: x position of the mouse
        :type xpos: float

        :param ypos: y position of the mouse
        :type ypos: float
        """

        # compute mouse displacement, save new position as last
        dx: float = xpos - self.lastx
        dy: float = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        # no buttons down: nothing to do
        if (not self.button_left) and (not self.button_middle) and (not self.button_right):
            return

        # get current window size
        width, height = glfw.get_window_size(window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

        mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

        # determine action based on mouse button
        if self.button_right:
            if mod_shift:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
            
        elif self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H

        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        # move camera according to mouse displacement
        mujoco.mjv_moveCamera(self.model, action, dx/width, dy/height, self.scene, self.camera)


    def _update_controls(self) -> None:
        """PID control toward the current joint target."""
        current_joint_pos: List[float] = self.data.qpos[:self.num_joints]
        target: List[float] = self._target

        # PID controller
        current_error: List[float] = target - current_joint_pos
        self.integral += (current_error + self.prev_error) / 2 * self.model.opt.timestep  # trapezoid rule
        self.integral = np.clip(self.integral, -2.0, 2.0)  # anti-windup clamp

        derivative: List[float] = (current_error - self.prev_error) / self.model.opt.timestep

        self.prev_error = current_error

        action: List[float] = (self.Kp * current_error) + (self.Ki * self.integral) + (self.Kd * derivative)

        assert(len(action) == self.num_joints) # result should be n-dimensional, address each joint

        self.data.ctrl = action
    

    def simulate(self) -> None:
        """ Runs the simulation & renders. This function will block until the window is closed.
        """

        # while the window is open, take simulator steps and render
        while not glfw.window_should_close(self.window):
            simstart: float = self.data.time

            # simulate enough steps to draw at the specified framerate
            while ((self.data.time - simstart) < (1.0 / self.framerate)):

                # step simulation
                mujoco.mj_step(self.model, self.data)

                # decide on next action
                self._update_controls()

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

            # Update scene and render
            mujoco.mjv_updateScene(self.model, self.data, self.viz_options, None, self.camera,
                                    mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
            mujoco.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        # close window
        glfw.terminate()


if __name__ == "__main__":
    import os
    scene_path = os.path.join(os.path.dirname(__file__), '..', 'mujoco_models/sawyer/computer_desk.xml')
    visualizer = MuJoCoVisualizer(scene_path)
    visualizer.simulate()
