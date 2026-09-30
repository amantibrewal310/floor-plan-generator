# Fix loop declaration

Written before the fix, and the prediction is not edited afterwards. Sections 1 to 3 are in the
commit that introduces this file, together with the `floorplan walkcheck` command that measures
the gate. Section 4 is added after the fix.

The benchmark here is the three LiDAR walkthroughs supplied as sample data (Stray Scanner
folders `c00a170fe1`, `1a8384c3f6`, `c7d28f72c6`). They have no tape measurements, so the gate
is one that needs none.

## 1. Worst gate

- **Gate:** the stitched plan must have every room placed and connected ("one whole-property
  floor plan ... with every room placed, connected and dimensioned"; "correct adjacency").
  Measured without tape: the phone never leaves the home, so **the share of the walk that falls
  inside some room** should be close to 100%, and **the rooms should form one connected plan**
  through their doorways (adjacency components = 1).
- **Failing number** (before run, the commit that adds this file):

  | capture | rooms | walk m | inside % | adjacent pairs | components |
  |---|---|---|---|---|---|
  | c00a170fe1 | 2 | 14.1 | 72.9 | 0 | 2 |
  | 1a8384c3f6 | 6 | 53.4 | 80.8 | 0 | 6 |
  | c7d28f72c6 | 5 | 97.8 | 59.5 | 0 | 5 |

  No two rooms are adjacent in any capture. The hallway connecting the rooms is missing from
  every plan, and in `c7d28f72c6` a whole room the phone walked around is missing too.
- **Gate threshold:** at least 95% of the walk inside a room, and components = 1, on each capture.

## 2. Root-cause hypothesis and evidence

- **Hypothesis:** the gap-closing step in `plan._enclose` fills the hallway as if it were wall.
  To close doorways, it bridges any gap up to 1.0 m (or 1.4 m) with a horizontal and a vertical
  line kernel. A doorway is a gap *in* one wall, but a hallway is the space *between* two
  parallel walls, about a metre apart. The kernel perpendicular to those walls bridges across
  the hallway and fills it solid. The hallway then never becomes a room, so the doors of the
  rooms that open onto it have no partner door, and adjacency (door faces that line up) finds
  nothing.
- **Evidence:**
  - Where the phone walked, the raw occupancy grid says "wall" at only 4%, 2% and 0% of camera
    positions (c7d28f72c6, 1a8384c3f6, c00a170fe1). After gap-closing, 42%, 23% and 23% of camera
    positions are on "wall". The closing, not the data, puts wall where the phone was. These
    numbers track the share of the walk outside every room (40%, 19%, 27%).
  - A picture of the closed wall mask for `c7d28f72c6` shows the hallway the camera walked back
    and forth along as solid wall, with the bedrooms enclosed around it.
  - It is not drift. With drift correction off (`floorplan walkcheck --no-drift-correction`),
    the inside share moves 59.5 → 71.4% on c7d28f72c6, 80.8 → 78.9% on 1a8384c3f6 and
    72.9 → 73.2% on c00a170fe1, and adjacent pairs stay at 0 everywhere. Drift handling does
    not bring the hallway back.
  - The positions-on-wall numbers above come from a diagnostic script run on the before commit;
    the gate numbers in section 1 come from `floorplan walkcheck`.

## 3. The fix and the predicted number

- **Change:** bridge gaps only *along* a wall. The horizontal line closing acts only on
  horizontally elongated wall cells (their run along the row is much longer than down the
  column), and the vertical closing only on vertically elongated ones. A doorway is a gap
  between two pieces of the same wall, so it is still bridged. The two parallel walls of a
  hallway are elongated along the hallway, so nothing bridges across it any more.
- **Predicted number after the fix:**
  - Camera positions on "wall" after closing: from 42 / 23 / 23% to at most 5% on each capture
    (what remains is doorway crossings).
  - Share of the walk inside a room: from 59.5 / 80.8 / 72.9% to at least 90% on each capture.
  - Adjacent pairs: from 0 to at least 1 on each capture. Components: from 5 / 6 / 2 to at most 2
    on each, since the hallway links the rooms that open onto it.
  - Main risk: a hallway that opens onto a space the phone never walked (seen only through a
    wide opening) stays unenclosed. The walk share then rises less on that capture, and it
    could miss the 95% gate even though the hallway is no longer filled.
  - The rooms that close today should keep their shape: every existing test must still pass,
    and the synthetic scenes must score no worse.

## 4. Result

Two attempts, both in the history.

**Attempt 1, the declared change** (commit `8b84ebc`): door-closing kernels skip cells of walls
running the other way.

| capture | inside % (before → after) | adjacent pairs | components |
|---|---|---|---|
| c00a170fe1 | 72.9 → 72.9 | 0 → 0 | 2 → 2 |
| 1a8384c3f6 | 80.8 → 83.8 | 0 → 0 | 6 → 5 |
| c7d28f72c6 | 59.5 → 59.8 | 0 → 0 | 5 → 5 |

The prediction was wrong. Camera positions on "wall" after closing stayed at 29 to 38% on
c7d28f72c6, against a predicted 5% at most. The hypothesis was right about *what* fills the
hallway (the closing step, not the data) but wrong about *how*. On real scans the wall mask is
full of 0.5 to 1 m blobs (clutter, noisy depth near walls). They are not clean elongated walls,
so the rule let them bridge across the hallway anyway. Without closing, the hallway is open, but
it leaks outside through the doorways it connects.

**Attempt 2, what shipped** (commit `366fa60`). The door-closing is left exactly as before.
Instead, for a walkthrough, the space that is free in the scan itself, inside the home, not
already a room and walked by the phone becomes a room of its own. A door also links its room to
the room whose outline lies within 0.5 m on the other side, since a hallway's door jambs are
often not seen.

| capture | inside % | adjacent pairs | components | gate (≥ 95% and 1 component) |
|---|---|---|---|---|
| c00a170fe1 | 72.9 → **92.6** | 0 → **2** | 2 → **1** | fail (inside 92.6) |
| 1a8384c3f6 | 80.8 → **96.6** | 0 → **5** | 6 → **3** | fail (3 components) |
| c7d28f72c6 | 59.5 → **86.2** | 0 → **1** | 5 → 7 | fail |

- **Against the prediction:**
  - At least 90% of the walk inside a room: met on 2 of 3.
  - At least 1 adjacent pair on each capture: met on 3 of 3.
  - At most 2 components on each capture: met on 1 of 3.
  - The gate itself still fails on all three. This is meaningful movement below the gate, not a pass.
- **Why it fell short, from the plans:**
  - On c7d28f72c6 the hallway comes back as two pieces, split where clutter narrows it. The
    stretch between the middle bedroom and the two rooms on the right is still missing, so those
    rooms stay unlinked.
  - Walked-space outlines are ragged: their edges are not snapped to the wall directions the
    way enclosed rooms are.
  - Those two are the next fixes. Snap walked-space outlines to the rooms' wall frame, and merge
    walked pieces that meet at a narrow neck.
- **Nothing that was right got worse:**
  - Every room the three sample plans had before is unchanged: area, walls with intervals,
    openings and ceiling.
  - All 14 synthetic cases are identical.
  - Photo folders are unaffected.
  - A new synthetic scene, `floorplan synth hall stray` (two rooms off a 1 m hallway), locks
    this in as a test. Before the fix it gives 2 rooms, 42.9% of the walk inside a room and no
    adjacency; after, 3 rooms, 99.0% and one connected plan.

Diff: `git diff 72ab94c..366fa60 -- src/`

Both runs regenerate with:

```
git checkout 72ab94c && uv run floorplan walkcheck ~/Downloads/c00a170fe1 ~/Downloads/1a8384c3f6 ~/Downloads/c7d28f72c6
git checkout 366fa60 && uv run floorplan walkcheck ~/Downloads/c00a170fe1 ~/Downloads/1a8384c3f6 ~/Downloads/c7d28f72c6
```
