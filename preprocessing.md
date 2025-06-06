1. Load all raw windows
2. Apply `find_ambiguous_windows` → filter out unclear/mixed-label windows
3. Apply `one_label_per_window` → ensure each window has a clean label

4. Downsample MAG → 41 samples
5. Extract MFCCs from MIC → (41, 13)
6. Combine ACC, GYR, MAG, MIC → (41, 22) per window
7. Create final dataset: X (7141, 41, 22), y (7141,)
8. Save the final dataset to disk


| Sensor | Windows | Samples | Notes                           |
| ------ | ------- | ------- | ------------------------------- |
| ACC    | 7141    | 41      | Already aligned (3-axis + time) |
| GYR    | 7141    | 41      | Already aligned (3-axis + time) |
| MAG    | 7141    | 62      | Needs downsampling to 41        |
| MIC    | 7141    | 3200    | Needs MFCC extraction (41×13)   |
