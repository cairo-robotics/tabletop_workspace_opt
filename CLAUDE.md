# CLAUDE.md - Sawyer MuJoCo + MoveIt Workspace

## Project Overview
This is a robotics simulation project using MuJoCo for physics,
MoveIt for motion planning, and RViz for visualization. The robot
is a Rethink Sawyer robot with a breakfast table scene containing
a cereal box, banana, bowl, and milk.

## MCP Servers Available
This workspace has the following MCP servers configured:

### MuJoCo MCP Server (`mujoco-mcp`)
- **Purpose:** Direct control of MuJoCo simulations through natural language
- **Capabilities:** Create/load simulations, step physics, set joint angles,
  load robots from MuJoCo Menagerie, query simulation state
- **Use this for:** Modifying sim parameters on the fly, stepping through
  physics, reading joint states and contact forces, testing grasp configurations
- **Robot models path:** We don't have this setup yet since we're only using Sawyer robot for now.

## Debugging Workflow

### Quick State Check (use MuJoCo MCP — no screenshots needed)
For non-visual debugging, prefer the MuJoCo MCP server to query state directly:
1. Query current joint positions via mujoco-mcp to get the Sawyer's joint angles
2. Compare against the expected MoveIt planned trajectory
3. Check contact forces on grasped objects to verify grasp stability
4. Read object poses (position + quaternion) from the sim state

### Visual Debugging (use claude-vision screenshots)
When debugging visual/spatial issues that require seeing the scene:
1. Take a screenshot of the MuJoCo renderer window (title: "{SCENE NAME}")
2. Take a screenshot of the RViz window (title: "moveit.rviz - RViz")
3. Compare object positions, robot joint configuration, and
   collision geometry between both views
4. Check /tf topic for frame transform issues

### Fallback Visual Debugging (if claude-vision MCP is unavailable)
Use the programmatic screenshot scripts:
```
python3 ~/sawyer_ws/src/tabletop_workspace_opt/mujoco_screenshot_server.py \
  ~/sawyer_ws/src/tabletop_workspace_opt/src/assets/scene_breakfast.xml \
  -o /tmp/mujoco_frame.png
```
Then view `/tmp/mujoco_frame.png`

For RViz:
```
python3 ~/sawyer_ws/src/tabletop_workspace_opt/rviz_screenshot.py /tmp/rviz_frame.png
```
Then view `/tmp/rviz_frame.png`

Always capture both views when debugging visual discrepancies.

## Key Commands
- Launch sim: `roslaunch tabletop_workspace_opt sim_moveit.launch`
- Check topics: `rostopic list | grep -E "joint_states|planning_scene"`
- Check TF: `rosrun tf tf_echo base right_hand`
- List MCP servers: `/mcp` (inside Claude Code)
- Dismiss ROS1 EOL popup after launch (blocks RViz interaction):
  ```bash
  sleep 5 && xdotool search --name "ROS End of Life" windowactivate --sync key Return 2>/dev/null &
  ```
  Run this right after launching. The `sleep 5` gives RViz time to open.
- Check topics: `rostopic list | grep -E "joint_states|planning_scene"`
- Check TF: `rosrun tf tf_echo base right_hand`
- List MCP servers: `/mcp` (inside Claude Code)
- **Pre-test cleanup** (always run before starting a new test session):
  ```bash
  pkill -f "mujoco" || true; pkill -f "rviz" || true; pkill -f "roslaunch" || true
  ```

## Typical Task Workflows

### Verifying a Grasp
1. Use mujoco-mcp to load the breakfast scene
2. Use mujoco-mcp to read the target object's pose (e.g., banana position and orientation)
3. Compare against the MoveIt grasp pose in the planning scene
4. If mismatch: check the MJCF XML for object placement and the SRDF for collision geometry
5. Use mujoco-mcp to step the sim after grasp execution and check contact forces
6. If grasp slips: increase object friction in the MJCF XML. If grasp goes into the object, you might need to decrease the object friction. 

### Debugging Sim-to-RViz Mismatch
1. Use mujoco-mcp to query all joint angles from the sim
2. Run `rostopic echo /joint_states -n 1` to get the ROS-published joint angles
3. If they differ: the mujoco-ros bridge node is likely not running or has a mapping error
4. Take screenshots of both windows to visually confirm the discrepancy

### Modifying the Scene
1. Edit the MJCF XML in `~/sawyer_ws/src/tabletop_workspace_opt/src/assets/`
2. Use mujoco-mcp to reload the scene and verify the changes took effect
3. Update corresponding collision objects in the MoveIt planning scene if object shapes changed

## Common Issues
- If objects appear in MuJoCo but not RViz: check that
  publish_planning_scene node is running
- If grasps fail: check collision geometry padding in
  moveit config (srdf) and the MJCF XML scene files. Modify object friction parameters.
- If mujoco-mcp reports stale state: the sim may need to be stepped forward
  (`mj_step`) before reading — a common gotcha after resetting joint angles
- If joint angles from mujoco-mcp don't match RViz: check joint name mapping
  between the MJCF model and the Sawyer URDF (MuJoCo uses `<joint name="...">`,
  ROS uses the URDF joint names — they must match or be remapped in the bridge node)

## MANDATORY: Cleanup After Testing
**ALWAYS run cleanup when you are done with a testing/debugging session.**
Leftover MuJoCo and RViz windows accumulate and waste resources. Do not
leave GUI windows open after you have finished your task.

### Close MuJoCo renderer windows
```bash
# Kill all MuJoCo viewer/renderer processes
pkill -f "mujoco" || true
# If the above doesn't catch the window, use wmctrl to target by title
wmctrl -c "{SCENE NAME}" 2>/dev/null || true
# Nuclear option if windows persist
pkill -f "simulate" || true
pkill -f "mujoco_viewer" || true
```

### Close RViz windows
```bash
# Kill RViz processes
pkill -f "rviz" || true
```

### Dismiss the ROS1 end-of-life popup in RViz
RViz shows a popup warning that ROS1 has reached its end of life.
This blocks interaction with RViz and must be dismissed:
```bash
# Use xdotool to find and close the popup dialog
xdotool search --name "ROS End of Life" windowactivate --sync key Return 2>/dev/null || true
# Alternative: dismiss any dialog in the foreground of the RViz window
xdotool search --name "moveit.rviz - RViz" windowactivate --sync key Escape 2>/dev/null || true
```
If xdotool is not installed: `sudo apt install xdotool`
If wmctrl is not installed: `sudo apt install wmctrl`

### Kill simulation server and planner processes
Each `roslaunch` spawns many child processes that are NOT killed by
`pkill -f roslaunch` alone — they keep running as orphans.
Multiple launches accumulate orphans quickly (can reach 100+ processes).
```bash
pkill -f "simulation_server.py" || true
pkill -f "move_to_cartesian_pose.py" || true
pkill -f "relaxed_ik_rust.py" || true
pkill -f "moveit_ros_move_group/move_group" || true
pkill -f "rosout/rosout" || true
pkill -f "static_transform_publisher" || true
pkill -f "robot_state_publisher" || true
```

### Full teardown (sim + MoveIt + RViz all at once)
```bash
# Kill the entire launch group cleanly
rosnode kill -a 2>/dev/null || true
pkill -f "roslaunch" || true
pkill -f "rviz" || true
pkill -f "mujoco" || true
pkill -f "simulation_server.py" || true
pkill -f "move_to_cartesian_pose.py" || true
pkill -f "relaxed_ik_rust.py" || true
pkill -f "moveit_ros_move_group/move_group" || true
pkill -f "rosout/rosout" || true
pkill -f "static_transform_publisher" || true
pkill -f "robot_state_publisher" || true
```

### When to clean up
- After completing any grasp test
- After finishing a debugging session
- Before starting a new test (to avoid stale windows from the previous run)
- If asked to "run tests" or "try this grasp," always clean up afterward
  unless the user explicitly says to keep windows open

## File Structure
- `~/sawyer_ws/src/tabletop_workspace_opt/src/assets/` - MJCF XML scene files
- `~/sawyer_ws/src/sawyer_moveit/sawyer_moveit_config/` - MoveIt configuration package
- `~/sawyer_ws/src/tabletop_workspace_opt/launch/` - ROS launch files
- `~/mujoco_menagerie/` - MuJoCo Menagerie robot models (optional, for loading alternative robots)