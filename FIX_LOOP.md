# Fix loop declaration

Filled in after the first full run of `floorplan bench benchmark/manifest.json` on real captures.
It is written before the fix, and the prediction is not edited afterwards.

## 1. Worst gate

- Gate:
- Failing number (before run, commit `____`):
- Gate threshold:

## 2. Root-cause hypothesis and evidence

- Hypothesis:
- Evidence (plots, per-capture numbers, a controlled experiment):

## 3. The fix and the predicted number

- Change:
- Predicted number after the fix:

## 4. Result

- After run (commit `____`):
- Moved from fail to pass? If not, why it fell short:
- Diff: `git diff <before>..<after> -- src/`

Both runs regenerate with:

```
git checkout <before> && uv run floorplan bench benchmark/manifest.json -o benchmark/out/before
git checkout <after>  && uv run floorplan bench benchmark/manifest.json -o benchmark/out/after
```
