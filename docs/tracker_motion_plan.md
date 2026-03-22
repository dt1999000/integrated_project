### Simple Motion-Aware Clustering Plan

This document specifies how to use simple 3D motion (centroid displacement between frames) and bag frequency to help clustering:
- **Reject clusters** that move or morph too much between frames.
- **Prefer clusters** whose centroids stay close to the previous centroid of the same object.

---

### 1. New data fields (minimal)

- **Per object / track (can be stored either inside `TrackedObject` or separately)**:
  - `last_center_3d: np.ndarray`  
    - 3D centroid of the object in the **previous frame** (LiDAR/world coords).
  - `last_frame_index: int`  
    - Frame index where `last_center_3d` was observed.

- **Global / configuration**:
  - `bag_freq_hz: float`  
    - Sampling frequency of the sequence (Hz). Default `45.0` for rosbags unless overridden.
  - `class_max_speed_mps: Dict[str, float]`  
    - Per-class hard maximum speed (m/s), used to derive max allowed centroid displacement per frame.

---

### 2. Rosbag frequency, filtering, and Δt

- **Raw bag frequency**:  
  - `bag_freq_hz = 45.0` for sequences loaded from rosbag.
  - This is the rate at which images would arrive **before any filtering/subsampling**.

- **Effective frame spacing after filtering**:
  - If you keep only a subset of frames (e.g. every 2nd or 3rd image), the **frame index gap** `n = f_k - f_{k-1}` already captures this:
    - If you keep 1 of every 3 frames, then `n = 3` between kept frames.
  - Time between two kept frames:
    - \(\Delta t_k = n / \text{bag\_freq\_hz}\).
  - This automatically **scales up** the allowed displacement when you skip frames.

- **Usage (only displacement, no full velocity model)**:
  - When comparing centroids between two **kept** frames `f_{k-1}` and `f_k`:
    - Frame gap: \(n = f_k - f_{k-1}\) (includes skipped frames).
    - \(\Delta t_k = n / \text{bag\_freq\_hz}\).
    - Observed displacement: \(\delta_k = \|p_k - p_{k-1}\|\).
  - Use \(\Delta t_k\) and class max speed to define **max allowed displacement** for gating:
    - \(\delta_{\max} = v_{\max} \cdot \Delta t_k\).

---

### 3. Common-sense speed priors (KITTI-like, simplified)

Approximate physical **hard max speed** per class (`class_max_speed_mps`); tune as needed:

- `car`: 40.0  (≈144 km/h)
- `truck/bus`: 35.0
- `bicycle`: 15.0
- `motorcycle`: 25.0
- `person`: 5.0
- `default`: 50.0  (catch-all upper bound)

For a frame gap \(n\) and `bag_freq_hz`:

- Max allowed centroid displacement:
  - \(\delta_{\max} = v_{\max} \cdot \frac{n}{\text{bag\_freq\_hz}}\).

Example with `bag_freq_hz = 45.0` and consecutive frames (`n = 1`):

- `car`:  
  - \(\delta_{\max} \approx 40 / 45 \approx 0.89\ \text{m}\).
- `person`:  
  - \(\delta_{\max} \approx 5 / 45 \approx 0.11\ \text{m}\).

---

### 4. Simple motion-based centroid check

When you have an object in frame `f_{k-1}` with centroid `p_{k-1}` and a **candidate cluster** in frame `f_k` with centroid `p_k`:

1. **Compute frame gap and Δt**  
   - \(n = f_k - f_{k-1}\) (usually `1` in your rosbag).
   - \(\Delta t_k = n / \text{bag\_freq\_hz}\).

2. **Compute observed centroid displacement**  
   - \(\delta_k = \|p_k - p_{k-1}\|\).

3. **Compute max allowed displacement for the object’s class**  
   - Look up `v_max` from `class_max_speed_mps` for the object label (or `default`).
   - \(\delta_{\max} = v_{\max} \cdot \Delta t_k\).

4. **Hard rejection rule**  
   - If \(\delta_k > \delta_{\max}\), **reject this cluster** as a valid continuation of that object:
     - “This cluster would require the object to move faster than physically allowed.”

5. **Preference rule between multiple clusters**  
   - If several clusters are not rejected (all with \(\delta_k \le \delta_{\max}\)):
     - Prefer the cluster with the **smallest** \(\delta_k\) (closest centroid to previous one).
   - This is enough to:
     - Avoid clusters that “jump” or “morph” to other objects.
     - Bias towards spatially consistent cluster assignments.

---

### 5. Where to apply this in the pipeline

The goal is **not** to rewrite the whole tracker, just to use motion to help clustering:

- After you have:
  - Clusters for each SAM mask in frame `k`.
  - For each object (track) you know which mask/cluster it used in frame `k-1`, with centroid `p_{k-1}`.

- For each object in frame `k-1`:
  1. Collect candidate clusters in frame `k` that belong to the **same mask / class**.
  2. For each candidate cluster:
     - Compute `δ_k` and compare to `δ_max` as in section 4.
     - Reject candidates with `δ_k > δ_max`.
  3. If at least one candidate remains:
     - Select the cluster with minimal `δ_k` as the **preferred continuation**.
  4. If none remain:
     - Treat the object as either:
       - Temporarily lost / occluded for this frame, or
       - Beginning of a new object if a very different cluster appears later check using the feature patch in tracking.py.

This keeps the tracker logic simple while using 3D motion to:
- Reject clusters that “jump” to another object.
- Prefer spatially consistent clusters when multiple options exist.

---

### 6. UI: bag frequency textbox

- **Location**:
  - Sidebar on `1_Dataset_Extraction` after choosing to load batch for detection, if the flag is True (where batch processing is configured).

- **Field**:
  - Label: **“Bag Frequency (Hz)”**.
  - Type: numeric text input or slider.
  - Default: `45.0`.
  - Valid range: `[1.0, 200.0]`.

- **Plumbing**:
  - Store in `st.session_state.params['bag_freq_hz']`.
  - When loading a rosbag, if its metadata contains the true frequency, set:
    - `st.session_state.params['bag_freq_hz'] = detected_freq`.
  - When instantiating `ObjectTracker` in batch mode:
    - `tracker = ObjectTracker(bag_freq_hz=st.session_state.params.get('bag_freq_hz', 45.0))`.

---

### 7. How this helps misalignment and occlusion

- In crowded scenes with occlusion, SAM masks and 3D clusters can swap between objects.
- This simple motion check:
  - Rejects clusters whose centroids move farther than physically possible between frames.
  - When multiple clusters are plausible, prefers the one with the **smallest centroid displacement** from the previous frame.
  - Does this without changing the core appearance-based assignment logic, just adding a **physics-based filter** around clustering.

