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

(After the fix.)

Both runs regenerate with:

```
git checkout <before> && uv run floorplan walkcheck <sample folders>
git checkout <after>  && uv run floorplan walkcheck <sample folders>
```
