# Thin Layer Placement Controller

This controller is intended for sandwich-style assembly with thin pieces that
need guarded vertical placement instead of a single fixed destination pose.

## What It Does

The controller runs this sequence:

1. move to hover over the stack anchor
2. move to a slow-approach height a few millimeters above the predicted stack
3. descend in guarded z steps
4. detect contact using low observed z progress
5. release the gripper
6. retreat vertically

The contact heuristic is force-free. It assumes contact when repeated z
commands no longer produce enough actual downward motion.

## Circular Grasp Assumption

This controller is paired with a SAM-based sandwich candidate pipeline.
The intended grasp heuristic is:

- pieces are circular or near-circular
- piece radius is smaller than the gripper opening
- grasp strategy is uniform `top-down circular grasp`
- the grasp point is the mask center
- the gripper yaw can be kept fixed because rotational symmetry makes in-plane
  orientation largely irrelevant

This means circular bread pieces, beef patty, tomato, onion, and cheese can all share the same
high-level pickup family even if their placement thickness differs.

## Command Format

Commands are JSON strings published to `/sandwich_assembly/command`.

### Start placement for the currently selected object

```json
{"action":"place_from_selected","stack_id":"default"}
```

### Start placement for an explicit grasp label

```json
{"action":"place","stack_id":"default","grasp_label":"ham_grasp"}
```

### Reset the current stack height model

```json
{"action":"reset_stack","stack_id":"default"}
```

### Update the stack anchor pose

```json
{
  "action":"set_stack_anchor",
  "stack_id":"default",
  "anchor_pose":{
    "position":[0.62,0.0,0.035],
    "orientation":[1.0,0.0,0.0,0.0]
  }
}
```

## Recommended Use Right Now

1. Keep target selection in the existing user-study pipeline.
2. Let a SAM-based top-down sandwich grasp pipeline complete the pickup.
3. Trigger this controller after grasp completion.
4. Treat `release_complete` as the success event for sandwich assembly steps.
