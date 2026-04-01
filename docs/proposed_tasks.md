# Proposed Shared Autonomy Tasks

## Available Grippable Objects (fit 59mm gripper)

| Object | Scene | Dimensions | Grip Width |
|--------|-------|-----------|-----------|
| apple | kitchen | 56x56x56mm | 56mm |
| banana | breakfast | 180x40x50mm | 40mm |
| bottle | kitchen | 50x50x190mm | 50mm |
| can | kitchen | 54x54x110mm | 54mm |
| cereal | breakfast | 172x46x306mm | 46mm |
| mug | desk | 56x56x110mm | 56mm |
| pen_cup | desk | 50x50x100mm | 50mm |
| sponge | kitchen | 100x50x40mm | 50mm |

Non-grippable (used as landmarks): bowl, book, phone, cutting_board, stapler, milk_carton

---

## Proposed Tasks (by complexity)

### Task 1: Kitchen Prep (4 objects, 8 steps, 3-4 choices per pick)
**Scene:** kitchen_prep
**Objects to manipulate:** apple, can, bottle, sponge
**Landmarks:** cutting_board

Steps:
1. Pick one of {apple, can, bottle, sponge} → 4 choices
2. Place on cutting board
3. Pick one of remaining 3 → 3 choices
4. Place on cutting board
5. Pick one of remaining 2 → 2 choices
6. Place on cutting board
7. Pick last one → 1 choice
8. Place on cutting board

**Why interesting:** 4 pick choices at initial state (highest disambiguation challenge).
Long horizon (8 steps). Tests how well the optimizer spreads 4+ objects.

### Task 2: Meal Assembly (cross-scene, 5 objects, 10 steps, 3-5 choices)
**Scene:** New combined scene with breakfast + kitchen objects
**Objects:** cereal, banana, apple, can, bottle
**Landmarks:** bowl, cutting_board

Steps:
1. Pick one of {cereal, banana, apple, can, bottle} → 5 choices
2. Place cereal near bowl / banana in bowl / apple on board / etc.
3. Continue until all placed

**Why interesting:** 5 simultaneous pick choices — maximum disambiguation
challenge. Would require creating a new combined scene XML.

### Task 3: Sort by Zone (3 objects, 6 steps, 3 choices, conditional goals)
**Scene:** kitchen_prep
**Objects:** apple, can, sponge
**Zones:** "food zone" (near cutting board), "cleaning zone" (far corner)

Steps:
1. Pick one of {apple, can, sponge} → 3 choices
2. Place food items near cutting board, sponge to cleaning zone
3. Pick next → 2 choices
4. Place appropriately
5. Pick last → 1 choice
6. Place appropriately

**Why interesting:** Different objects go to different destinations.
Tests inference when goals are spatially diverse (food vs cleaning zone).

### Task 4: Stacking / Nesting (2-3 objects, 4-6 steps, order matters)
**Scene:** desk
**Objects:** mug, pen_cup
**Task:** Stack pen_cup inside/near mug, both near phone

Steps:
1. Pick mug or pen_cup → 2 choices
2. Place near phone
3. Pick the other → 1 choice
4. Place near/on top of first

**Why interesting:** Order matters — you must place the base object first.
Tests task dependency inference.

### Task 5: Kitchen Cleanup (4 objects, 8+ steps, mixed destinations)
**Scene:** kitchen_prep
**Objects:** apple, can, bottle, sponge
**Destinations:** cutting_board (food items), far corner (cleaning), near edge (drinks)

Steps:
1. Pick any of 4 → 4 choices
2. Place apple on cutting_board
3. Pick any of 3 → 3 choices
4. Place can near edge (drinks zone)
5. Pick any of 2 → 2 choices
6. Place bottle near can
7. Pick sponge → 1 choice
8. Place sponge in cleaning zone

**Why interesting:** Multiple destination zones. Some goals cluster
(can + bottle go to same zone), making inference harder.

---

## Recommended Implementation Order

1. **Kitchen Prep** — Uses existing scene, 4 grippable objects, straightforward
2. **Sort by Zone** — Same scene, tests spatial goal diversity
3. **Meal Assembly** — Requires new combined scene (most work)

## What I Can Create Without New Assets

Tasks 1, 3, 4, 5 use existing scenes and objects.
Task 2 needs a new scene XML combining breakfast + kitchen objects.

## What Would Need Online Assets

For even more variety (6+ objects), we could add:
- Colored blocks (simple box geoms, no mesh needed)
- Cups/glasses (cylinder geoms)
- Plates (flat cylinder geoms)

These are trivial to add as primitive MuJoCo geoms without any mesh files.
