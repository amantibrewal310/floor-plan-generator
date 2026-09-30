# Capture protocol

Follow this page exactly. It works with any iPhone 15 or newer. Nothing needs printing and nothing
needs installing, except one free app for the LiDAR tier.

## Before you start (all tiers)

- Turn on every light in the rooms. Open curtains and blinds.
- Close mirrored wardrobe doors if you can. Never aim the camera straight at a mirror.
- Leave every door between rooms fully open.
- Hold the phone in **landscape** (sideways), on the normal **1x** lens. Do not use 0.5x or zoom.
- Keep people and pets out of the shot.

## Tier 1: photos (any iPhone 15 or newer)

App: the built-in Camera app, Photo mode.

For each room, including hallways and landings:

1. Stand in a corner with your back to it, phone at chest height.
2. Aim at the opposite corner. The floor line and the ceiling line must both be in the picture.
   Tilt the phone up a little if the ceiling is cut off. Take one photo.
3. Repeat in every corner. Four corners means four photos.
4. For each door out of the room, step back about 2 m and take one photo that shows the whole
   door frame, both sides and the wall above it.
5. Stay between 2 and 8 photos per room. If a room has more than four doors, drop door photos first.

Don't take close-ups, don't stand in doorways, and don't shoot portrait.

Hand-off: AirDrop the photos to the Mac. Make one folder per room, named after the room
(`kitchen`, `hall`, `bedroom-1`). The folder name becomes the room's name on the plan.

```
uv run floorplan photos kitchen hall bedroom-1 -o out
```

## Tier 2: video (any iPhone 15 or newer)

App: the built-in Camera app, Video mode, 1x lens, landscape.
Settings > Camera > Record Video: 1080p at 30 fps. 4K works too, it's just slower to process.

1. Start recording in the first room. One continuous clip covers every room.
2. In each room, walk slowly around it about 1 m from the walls. Point the camera across the room
   at the far wall, never at the wall right next to you. Keep the floor line and the ceiling line
   in view. Plan on 30 to 45 seconds per room.
3. Walk through each doorway facing forward, at half your normal pace.
4. If you can, finish where you started.
5. Walk slowly and turn slowly. A blurry frame is a lost frame.

Hand-off: AirDrop the video to the Mac.

```
uv run floorplan video IMG_1234.MOV -o out
```

## Tier 3: LiDAR (iPhone 15 Pro / Pro Max, 16 Pro / Pro Max, 17 Pro / Pro Max)

App: **Stray Scanner** (free, App Store, by Stray Robots). Allow camera access when asked.

1. Open Stray Scanner and tap the record button.
2. Walk exactly as in the video tier: slowly, 1 m from the walls, camera aimed across the room,
   floor and ceiling lines in view, every room in one recording.
3. Tap stop. Don't cover the camera or the LiDAR sensor (the small black circle next to the lenses).

Hand-off: in the iPhone **Files** app, open On My iPhone > Stray Scanner. Long-press the newest
folder, tap Compress, and AirDrop the `.zip` to the Mac. Unzip it. If the folder isn't in Files,
plug the iPhone into the Mac, open Finder, select the iPhone, go to the Files tab, and drag the
Stray Scanner folder out.

```
uv run floorplan lidar 71de12f9 -o out
```

## What you get

`out/plan.svg` is the drawing and `out/plan.json` is everything, with a 90% interval on every
number. The run also prints a short summary. If a room is missing from the plan, the walls were
not captured all the way round. Look at `warning:` lines in the output, then recapture that room.

## Optional: a printed marker (photo and video tiers)

If a printer is handy, `uv run floorplan markers` writes A4 marker sheets. Print one at 100% scale,
measure its black square in metres, and lay it flat on the floor where the camera will see it.
Then add `--marker-size 0.18` (your measurement) to the command. The marker fixes the scale and
the intervals tighten. Without it, the scale comes from the model and the intervals stay wide.
