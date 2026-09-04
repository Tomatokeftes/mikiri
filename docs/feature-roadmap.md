# Feature roadmap: what a self-coaching player actually needs

This page is the output of a research pass over badminton coaching syllabi
(BWF Level 1 and 2, Shuttle Time, UKCC Level 2), the time-motion and
performance-norm literature, the motor-learning work on how feedback should
be delivered, the standard solo drill protocols, and what existing racket-sport
analysis products ship. It is written as a build plan, ranked by value against
effort, and it is as much about what Mikiri should refuse to build as what it
should build.

Two framing decisions run through all of it.

**The unit of analysis is the movement cycle, not the stroke.** BWF teaches
movement as START, APPROACH, HIT, RECOVER, and each cycle is triggered by the
opponent striking the shuttle. A tool that only looks at the moment of contact
is looking at one quarter of what a coach watches.

**The primary comparator is the player's own previous session.** Elite figures
appear as labelled context and never as targets: elite contact height and jump
height are driven by stature and jumping ability, not by technique quality.

## Ranked roadmap

### 1. Camera-view classifier and per-metric view gating

`must-have` &middot; `small effort` &middot; `all`

**How.** geometry.py already recovers full extrinsics from the 4 corners plus the two net-post tops. Add view_class(camera) that computes camera azimuth in court coordinates (atan2 of camera ground_point relative to court centre, against the court long axis) and elevation. Classify SIDE_ON (within 30 deg of the net-parallel axis), REAR (within 30 deg of the long axis), OBLIQUE otherwise. Add a static allow-list keyed on view class and consult it in coaching.assess_trust: image-plane angles (trunk lean, knee angle, contact in-front offset) require SIDE_ON; stance width, landing width, across-body offset, any frontal-plane alignment require REAR; everything derived from the homography plus extrinsics (court position, distance, speed, zone time, contact height in metres, jump clearance, split-step presence) is view-robust and never gated. Gated metrics are not computed, not greyed out with a number behind them.

**Bands.** SIDE_ON = |camera azimuth - 90 deg| <= 30; REAR = |azimuth| <= 30; OBLIQUE otherwise. Recommended rear-view filming spec, from the IJSPT frontal-plane work: camera behind the baseline on the centre line, >= 0.8 m height, perpendicular to the landing surface. No user baseline needed - this is geometry, not a norm.

**UI.** A one-line badge under the video player: 'Side-on view - 14 of 21 metrics available'. Clicking it opens a panel listing each hidden metric with one sentence of reason and a small court diagram showing where to put the phone for it. The same badge string goes into the export pack header so a screenshotted number always carries its view.

### 2. Rally and work-bout segmentation (solo-safe)

`must-have` &middot; `medium effort` &middot; `all`

**How.** Everything downstream is currently whole-clip: court.footwork_metrics aggregates the entire track. Add a segmentation pass producing a list of bouts. Two modes, chosen automatically and stated in the UI. RALLY MODE (opponent present): bout start = first detected contact after a quiet period, bout end = last contact before the next quiet period, where quiet = player speed below 0.5 m/s for more than 3 s. SHADOW MODE (solo drills, the realistic input): bout = any run of >= 3 s where speed exceeds 0.5 m/s, bounded by >= 3 s of stillness - no contacts required. Drop gaps longer than 60 s entirely (walked off, paused filming) and report how many were dropped. Refactor footwork_metrics to take a frame slice so every existing metric becomes per-bout for free, and keep the whole-clip figure as an aggregate.

**Bands.** Elite reference, labelled as such: men's singles rally 8.8 +/- 2.0 s, rest 22.9 +/- 10.6 s, work density 0.4 +/- 0.1, effective playing time 34.1 +/- 5.3 %; women's singles rally 9.3 +/- 2.0 s, rest 17.2 +/- 4.4 s, EPT 29.0 +/- 6.6 %; men's doubles rally 6.5 +/- 0.7 s. Junior/club rally 4.6-7.0 s. For shadow mode there is no published band - work interval and rest are needs user baseline, compared against the player's own earlier sessions.

**UI.** A bout strip under the timeline: one bar per rally or work interval, width = duration, gap = rest, coloured by mean speed. Clicking a bar scrubs to it and filters every metric panel to that bout. A summary row above it: 'shadow mode, 14 work intervals, median 11.2 s work / 24.0 s rest'.

### 3. Per-bout distance, metres per minute and metres per shot

`must-have` &middot; `small effort` &middot; `all`

**How.** Once bouts exist this is a re-slice of court.build_track plus footwork_metrics. Per bout: path length of the smoothed court-plane track (keep the existing smoothing - unsmoothed keypoint jitter inflates distance by 10-20 %), duration, mean speed (distance/duration over live frames only), peak speed as the 98th percentile that footwork_metrics already uses. Report distance per total elapsed minute AND distance per minute of live bout time, labelled separately, because mixing them is the commonest way this number is quoted wrongly. Metres per shot = bout distance / own contacts in the bout, from strokes.py; publish the own-contact variant only, and say so, because the two-player figure is not measurable solo.

**Bands.** Elite reference: 44.6 +/- 1.6 m/min (MS) and 43.5 +/- 0.9 m/min (WS) over total match time; world-top per-rally distance 16-25 m (men) and 21-22 m (women); peak speed 10.9 +/- 0.1 km/h (MS), 10.7 +/- 0.3 km/h (WS); mean in-rally speed 2.2-2.6 m/s at world level. National-squad drills run at 65-76 m/min, i.e. 1.5-1.7x match rate - the right comparator for a shadow session. Metres per shot is derived, not measured, at roughly 4.7-5.4 m per own contact in elite singles; present it as derived. Never mix IMU-derived distance (703.6 +/- 181.9 m per set) with video homography distance - different scales.

**UI.** A three-tile row at the top of the footwork tab: 'Distance 312 m', 'Rate 58 m/min (drill band 65-76)', 'Per shot 4.1 m'. Under each tile, the per-bout distribution as a small strip plot with individual bouts clickable. Uncertainty from distance_uncertainty_m printed as a +/- on the distance tile, never hidden.

### 4. Split-step presence rate and split-to-first-step latency (no opponent required)

`must-have` &middot; `medium effort` &middot; `beginner`

**How.** movements.py already has a low-threshold split-step hop detector. Promote it to a first-class per-clip event stream and harden it: candidate window 400 ms, requiring (a) mid-hip rise then fall of >= 2 cm in metres via the recovered camera geometry, (b) both ankles above their rolling 1 s baseline by > 1.5 cm for >= 2 consecutive frames, (c) ankle separation widening >= 8 % and staying widened >= 100 ms after touchdown. Because a solo player has no opponent to anchor to, report only self-contained quantities: presence rate against direction-change events rather than opponent hits (a split step should precede each change of travel direction in shadow work), split-to-first-step latency = time from landing frame until hip horizontal speed exceeds 0.5 m/s, landing stance width divided by standing body height, hip rise in cm, and distance from base at the split. Refuse the whole feature below 50 fps rather than reporting a false 0 % - at 30 fps the flight phase is 1-3 frames.

**Bands.** Elite reference (BWF World Championships 2023, homography video, three world-top players): split-step reaction time 0.24 +/- 0.05 s, stance width 0.51-0.52 of body height, split position 42-55 cm from base. Amplitude: coaching guidance is 1-3 cm of air; higher than a racket grip width (~3 cm) is inefficient. Presence rate bands are a product heuristic and must be labelled so: < 50 % not yet a habit, 50-85 % inconsistent, > 85 % habitual. Ground contact time: needs user baseline.

**UI.** A dedicated card on the movement tab: a big percentage, the rep count next to it ('split step on 9 of 14 direction changes'), and a strip across the clip timeline showing where the split steps stop happening (usually late in a bout). Each detected split is clickable to scrub. Latency and stance width live in the detail view, not the headline.

### 5. Distribution-first metric storage and minimum rep counts

`must-have` &middot; `medium effort` &middot; `beginner`

**How.** This is an architectural change, not a metric: today movements.py computes a value per event and coaching.metric_values collapses them. Store the full per-rep vector for every event metric, with its frame index, side label, keypoint confidence and view-gate result attached. Compute median, IQR and a consistency score = 1 - (IQR / reference spread), where reference spread is the player's own first recorded session where one exists and a fixed per-metric spread otherwise. Enforce a minimum rep count before any aggregate is displayed at all, and drop reps below the confidence threshold rather than averaging them in - reporting how many were dropped.

**Bands.** Minimum reps before an aggregate is shown: 8 for stroke metrics, 15 for footwork metrics. Below that, show the individual reps only and say 'based on 5 overheads - not enough for a summary'. Consistency itself is needs user baseline by construction: the comparison that matters is the same metric in the player's own previous session.

**UI.** Every metric tile shows median, IQR and rep count on the face - 'Lunge depth 0.61 m, spread 0.14 m, 11 reps'. Under it a strip plot of individual reps, each dot clickable to scrub to that frame, outliers visibly separated. Aggregates below the rep minimum render as a greyed tile with the rep count and no number.

### 6. Shadow-footwork protocol mode

`must-have` &middot; `medium effort` &middot; `all`

**How.** Define one fixed, repeatable five-minute filming protocol and build a mode around it: a six-corner shadow sequence (rear forehand, rear backhand, mid forehand, mid backhand, front forehand, front backhand) for a fixed number of cycles, then a fixed number of shadow overheads. Detect the sequence from the court-plane track by matching corner arrivals in order, segment it into named reps automatically, and lock the metric set to the ones that mean something in shadow work (split step, lunge, base recovery, time-to-turn, corner symmetry, speed CV, fatigue drift). Store each run as a session in project.py so compare_sessions has a like-for-like comparator. This is the single highest-leverage product decision for a solo user: it converts an uncontrolled clip into a repeatable measurement.

**Bands.** Everything in this mode is needs user baseline by design - the protocol exists so session N is comparable to session N-1. The one external anchor is drill work rate: national-squad drills run 65-76 m/min versus 44.6 m/min in match play, so a shadow session below match rate is a genuine finding.

**UI.** A mode selector on load: 'Match/rally clip' or 'Shadow protocol'. In protocol mode the app shows a printable one-page card of the sequence before filming, then after analysis a per-corner table (time to corner, lunge depth, recovery time, split step yes/no) and a session-over-session line for each. The default view after the second run is the delta, not the absolute.

### 7. Prioritiser: hard cap of three findings, with causal precedence

`must-have` &middot; `small effort` &middot; `all`

**How.** coaching.top_cues already caps cues; tighten it into the real thing. Score each candidate by (1) confidence - rep count, mean keypoint confidence, view-gate pass, error-ellipse size, all of which assess_trust already computes; (2) effect size - deviation from the band or from the player's own baseline expressed in units of the metric's own IQR, not raw units; (3) level-appropriateness - a lookup that promotes beginner-tagged findings for a beginner and suppresses advanced ones entirely; (4) causal precedence - a small hand-authored dependency graph so a cause outranks its symptom and the symptom is then suppressed from the list: backpedalling outranks square-on contact, missing/late split step outranks large recovery distance, shallow lunge outranks low net contact, fatigue drift outranks any single depth finding.

**Bands.** Hard cap of 3 headline findings (BWF Level 2 tells coaches to focus on 2-3 crucial points; the UKCC worked example reduces 14 observed faults to 4 goals). Self-video guidance: one attentional cue per viewing, clips of 10-20 s, at most two replays. These are the bands - they are about count, not magnitude.

**UI.** The findings panel shows exactly three cards and nothing else. Each card: one plain sentence, one number with its band and rep count, a 10-20 s clip anchored to both the worst and a representative-median rep, and one BWF-sourced effect cue ('imagine the ground is hot', 'balance a shuttle on your head'). Everything else sits behind a single 'more detail' control; the full metric set appears only in the export pack.

### 8. Court occupancy heatmap with corner asymmetry index

`high` &middot; `small effort` &middot; `all`

**How.** court._heat_layer and _coverage_shape already exist and already use the error ellipse as the kernel. Add the numbers that survive a screenshot: a signed rear-corner asymmetry index (t_racket_rear - t_nonracket_rear) / (t_racket_rear + t_nonracket_rear) and the same for the two front corners, plus the court-plane centroid offset from the geometric centre in metres, split into along-court and across-court components. Detect handedness once per clip from which wrist leads through the contact window, cache it, and report every side label as racket-side / non-racket-side rather than left/right so left-handers are not silently inverted.

**Bands.** Asymmetry index target |index| < 0.15, labelled explicitly as a product heuristic - no published occupancy norm for amateur badminton exists. Centroid offset: needs user baseline, plus the objective court anchor (singles 13.40 x 5.18 m) so 'your base sits 1.1 m behind centre' is interpretable without a norm. Require >= 6 reps on each side before showing an asymmetry figure.

**UI.** The existing court map gains a percentage table beside the picture (time per zone, rear racket-side vs rear non-racket-side) and a single headline sentence: 'You spend 41 % more time in the rear forehand corner than the rear backhand corner.' The far-court cells stay visibly smeared by the error-ellipse kernel; a caption says why.

### 9. Lunge quality pack: depth, full extension, count, rate and symmetry

`high` &middot; `medium effort` &middot; `intermediate`

**How.** movements.py already detects lunges and computes lead-knee plant, lunge length and hip drop as percentages. Convert to metres and add the missing pieces. Detect a lunge as: ankle separation in court metres above 1.2x leg length, mid-hip height below 80 % of standing, front-ankle horizontal velocity crossing zero. At minimum hip height compute front-knee angle (SIDE_ON only), step length in metres and normalised by leg length, hip drop in metres, and trunk lean from vertical corrected for known camera roll. Add a straight-leg-phase boolean: did front-knee angle exceed 165 deg at any frame between step-out and plant. Count lunges per bout and per minute and report the racket-side / non-racket-side split.

**Bands.** Peak front-knee flexion 81.7 +/- 8.6 deg unfatigued and 77.5 +/- 8.3 deg fatigued - labelled as 3D lab values, so expect 5-15 deg of projection error from a single side-on phone and show a band not a point. Lunge count per match per player: MS 158 +/- 50, WS 175 +/- 48, doubles 47-51, i.e. roughly 1.95 lunges/rally in MS and 0.55 in doubles. Symmetry: flag outside a 40:60 racket-side split. Trunk lean heuristic: < 20 deg upright, 20-35 deg acceptable under pressure, > 35 deg flag - and suppress the flag when the lunge was unusually deep, because BWF notes lunges deepen under pressure. Straight-leg-phase rate: needs user baseline.

**UI.** A lunge table with one row per rep (timestamp, side, depth in metres, knee angle, straight-leg phase yes/no, trunk lean), sortable, every row scrubbing to the frame. Above it three tiles: count, median depth with IQR, side split as a small horizontal bar. The knee-angle column carries a 3D-lab-reference footnote marker.

### 10. Contact point geometry: height as a percentage of standing reach

`high` &middot; `medium effort` &middot; `intermediate`

**How.** racket.py already estimates the racket head from forearm extension and the new camera geometry already gives contact height in metres. Add the normalisation that makes it interpretable without an elite comparison: contact_height / standing_reach, where standing reach is measured once from a calibration frame of the player reaching up, falling back to height x 1.21 + 0.665 m racket. Then build a local frame at the racket shoulder (x = court-plane direction toward far-court centre, y = across body) and report the signed in-front/behind offset in metres - gated to SIDE_ON - and the across-body offset gated to REAR. This is the clearest single case where view gating earns its keep, because a single camera cannot resolve depth along its own optical axis.

**Bands.** Primary band, self-normalised: 95-100 % of standing reach indicates a genuinely extended arm at contact; below about 85 % indicates a bent, dropped arm. Elite reference for context only, never as a target: contact height 2.90 m (men) vs 2.46 m (women), a gap driven by jump height (53.6 vs 14.5 cm) and stature rather than technique. In-front offset: slightly in front of the racket shoulder is the BWF target; behind is always a fault. Magnitude of 'slightly': needs user baseline.

**UI.** A per-overhead card showing contact height in metres rounded to 5 cm, the percentage-of-reach figure as the headline, and a small side-on stick figure with the contact point marked relative to the shoulder. The tile carries a permanent 'racket head is estimated from forearm extension' note - it is the roughest input in the app and must say so.

### 11. Side-on versus square-on at overhead contact

`high` &middot; `medium effort` &middot; `intermediate`

**How.** With full extrinsics, back-project both shoulder keypoints onto a horizontal plane at shoulder height (both shoulders at the same height is a good approximation at contact), giving a shoulder line in court metres. Shoulder azimuth = angle to the net-parallel direction, mod 180, signed by handedness. Compute it at the contact frame and at the frame of peak elbow height; the difference is the trunk rotation actually used through the hit. Repeat on the hip line for a 2D pelvis-thorax separation proxy. Suppress the metric when the shoulder line is short in pixels (nearly pointing at the camera - ill-conditioned) or when shoulder keypoint confidence is low, and attach the error-ellipse-derived confidence.

**Bands.** Product heuristic, labelled as such: contact azimuth > 60 deg = side-on, 30-60 deg = partially turned, < 30 deg = square-on (flag). Rotation used through the hit (preparation azimuth minus contact azimuth) below 15 deg suggests the arm swung without the trunk. No published azimuth band exists; the supporting evidence is directional only (elite men show greater trunk flexion and counter-rotation through the backswing than elite women).

**UI.** A small top-down body icon per overhead showing the shoulder line against the net line, with the number beside it. In the findings panel this appears as a symptom, so the prioritiser suppresses it whenever backpedalling is already the headline finding - the causal chain is shown as a sentence: 'You arrive at the rear court still facing the net, which is why contact is square-on.'

### 12. Backpedalling detection and time-to-turn

`high` &middot; `medium effort` &middot; `intermediate`

**How.** Over each rearcourt approach window (from split-step landing, or from the start of backward travel in shadow mode, to contact) compute mid-hip court-plane velocity and hip-line azimuth. Backpedalling = frames where the backward velocity component exceeds 0.8 m/s while hip azimuth is under 30 deg from square. Report the percentage of approach frames spent backpedalling, the count of approaches containing any, and time-to-turn: milliseconds from the start of the approach to the first frame where hip azimuth exceeds 45 deg. Add path curvature = travelled path length / straight-line distance from base to contact position, since BWF teaches an arcing approach.

**Bands.** Product heuristics, labelled: flag any approach with > 30 % backpedalling frames; target time-to-turn under 250 ms. Path curvature meaningfully above 1.0 indicates the taught arcing approach; about 1.0 indicates a straight backpedal. Objective anchor: base to rear corner is 4-5 m in singles and elite contact intervals are 0.8-1.2 s, so 300 ms spent not turning is a large share of the available time.

**UI.** On the court map, rearcourt approach paths are drawn as coloured tracks - the backpedalling segments in a distinct colour so the fault is visible as a shape rather than a number. Time-to-turn is a single millisecond figure with the per-approach distribution behind it.

### 13. Base position and recovery quality

`high` &middot; `small effort` &middot; `beginner`

**How.** footwork_metrics already computes base as the median position and mean/max distance from base. Make it event-anchored and adaptive. Base = the mode of the occupancy density rather than the median, recomputed as a running value so it stays the 'fluid position' BWF describes. For each contact or direction change, find the following local speed minimum and record distance from base at that minimum and time from the event to it. Add a 'caught in transit' counter: the number of events where hip speed exceeded 1.0 m/s at the moment the event occurred - the single most actionable number in the set, because its target is unambiguously zero.

**Bands.** Elite reference: world-top players split-step 42-55 cm from base. Recovery distance and recovery time are otherwise needs user baseline - no published metre band exists - so report the player's own median plus the percentage of events above their own 75th percentile. Caught-in-transit count: target 0, no baseline needed.

**UI.** The court map gains a base marker with a ring at the player's own median recovery distance, and the individual recovery end-points plotted as dots so the spread is visible. One sentence headline: 'You were still moving at 4 of 19 shots.'

### 14. Scissor-kick detection, landing width and time to first recovery step

`high` &middot; `small effort` &middot; `intermediate`

**How.** movements.py already has a scissor-kick smash profile. Add the two measurements the coaching literature actually names. Kick-through: over contact to contact+600 ms, check whether the ankle that was further from the net at contact becomes the nearer one - a sign flip in their net-ward ordering. Landing width: at the first frame after contact where both ankles return to baseline height, court-plane distance between ankle projections in metres, divided by shoulder width measured from the same pose. Time to first recovery step: frames from landing until hip velocity toward base exceeds 0.5 m/s.

**Bands.** Landing width heuristic, labelled: below 1.0x shoulder width = narrow (flag), 1.0-1.8x = stable. Crucial caveat to encode in the copy: elite women predominantly use a kick-through rather than a two-footed jump while elite men predominantly jump, so absence of a two-footed jump is not a fault and the tool must never imply it is. Time to first recovery step: needs user baseline.

**UI.** A per-overhead row in the same table as the lunge pack, with a small two-dot foot diagram showing landing width against a shoulder-width reference bar. No red styling.

### 15. Jump count, jump rate and landing absorption

`medium` &middot; `small effort` &middot; `intermediate`

**How.** The flight detector already exists. Count events where both ankles leave the floor plane for a minimum duration, deduplicated within a short window, split by the existing profile labels (jump smash, scissor-kick smash, block jump). Per bout and per minute. Landing: knee angle at the first ground-contact frame (SIDE_ON only), one- versus two-footed (both ankles reaching baseline within 100 ms), absorption depth = hip height at landing minus minimum hip height in the following 400 ms in metres, absorption time in ms, and a stiffness proxy = depth / time. Jump height in metres and contact clearance above the net are already available from the recovered geometry.

**Bands.** Elite reference: 0.9 +/- 0.0 jumps/min (MS) and 0.8 +/- 0.1 (WS) in match play - worth showing precisely because it tells an amateur that jumping is rare at the top; drills run at 6.7-8.8 jumps/min. Per match: MS 71 +/- 22 jumps, WS 58 +/- 18. Knee flexion at initial contact 16-24 deg in female smash landings, i.e. a fairly extended knee at touchdown is normal in this sport - do not flag it as stiff on its own. Absorption depth and time: needs user baseline. Net height 1.524 m at centre, 1.55 m at the posts.

**UI.** A count tile with the per-minute rate beside the elite rate, and a landing sub-panel per jump showing knee angle at contact, one/two-footed and absorption depth. Copy explicitly notes that an unanticipated landing loads harder than a planned one, so a stiff landing mid-rally may be surprise rather than habit.

### 16. Non-racket arm elevation through the overhead

`medium` &middot; `small effort` &middot; `beginner`

**How.** Over the window from 500 ms before contact to contact, track the non-racket wrist. Peak (wrist height - shoulder height) in metres; a pull-in feature = change in horizontal distance from the non-racket wrist to mid-hip between peak elevation and contact; and a robustness check = fraction of the window in which the non-racket wrist sits above the non-racket elbow. Racket side is already cached from the handedness detection.

**Bands.** Binary threshold, plainly interpretable and labelled a heuristic since no published norm exists: non-racket wrist peaks at or above shoulder height = arm used; below the shoulder = flag. Report as a count, not a percentage of a degree.

**UI.** A single sentence with a count on the overhead card: 'Front arm up on 4 of 12 overheads.' Clicking cycles through the 8 where it was not. No angle is shown at beginner level.

### 17. Speed variability index (movement composure)

`medium` &middot; `small effort` &middot; `advanced`

**How.** On the same smoothed metre-space speed series used for distance, compute CV = std(speed) / mean(speed) per bout, and plot it against bout length or shots per bout. Flag bouts where CV departs markedly from the player's own baseline in long bouts as candidate 'drifting' bouts and offer to scrub straight to them. Cheap - it is a second statistic over an array that already exists.

**Bands.** Directional only, and must be labelled so: world-top male players hold CV near 1.0 into the middle and late stages of rallies while high-school players' CV drifts as rallies lengthen, and top females' CV decreases gradually through long rallies. These came from boxplots, not tables, so no exact threshold is defensible. Primary comparison is needs user baseline.

**UI.** A scatter of CV against bout length with the player's own regression line, and last session's line ghosted behind it. Headline sentence when the pattern appears: 'In your long rallies you cover more ground but move less purposefully.'

### 18. Within-session fatigue drift

`medium` &middot; `small effort` &middot; `advanced`

**How.** Bin every per-rep metric by clip time into terciles and report tercile medians plus a slope with a confidence interval. Do the same for the work-rate proxies already computed: distance per bout, mean and peak speed, direction changes per bout, recovery distance. Raise a drift flag when the last tercile differs from the first by more than that metric's own within-tercile IQR. Then, and this is the point of the feature, make the finding text choose between two interpretations: if lunge depth and speed drift down together, call it conditioning; if depth drifts and speed does not, call it technique.

**Bands.** Published direction of drift, useful as a sanity check: lunge peak knee flexion falls from 81.7 +/- 8.6 to 77.5 +/- 8.3 deg under fatigue and leg stiffness rises from 9.03 +/- 2.33 to 10.49 +/- 3.65 BW/m; hip transverse-plane ROM falls from 27.0 to 23.0 deg. Magnitudes for an individual: needs user baseline. Match pacing context: elite MS rest 22.9 +/- 10.6 s at a work:rest of 0.4 +/- 0.1.

**UI.** A three-column tercile table for the top metrics with an arrow per row, and one interpretation sentence at the top. Fatigue drift outranks any single-rep depth finding in the prioritiser, so when it fires it usually takes a headline slot and suppresses the metric it explains.

### 19. Session-over-session comparison as the default view

`medium` &middot; `medium effort` &middot; `all`

**How.** coaching.compare_sessions and session_summary already exist; the missing part is making the delta the landing screen rather than a buried tab. Store each analysed clip as a session record keyed on the protocol used (shadow protocol runs compare only against other shadow protocol runs; free clips compare only against free clips, with a warning when bout definitions differ). Show change with an uncertainty band derived from the per-rep IQR and the position error, so a change smaller than the noise is drawn as 'no change' rather than as an improvement.

**Bands.** Needs user baseline throughout - this feature is the mechanism by which every other 'needs user baseline' band gets a comparator. Rule: a change is only reported as a change when it exceeds the metric's own within-session IQR.

**UI.** On opening a second or later session, the default panel is 'Since last time': at most five rows, each a metric name, the delta with its uncertainty, and a sparkline over all sessions. Absolute values are one click away. This is also the main defence against novelty decay - the second session must show something the first could not.

### 20. Rear-leg loading and hip drive before the overhead

`medium` &middot; `medium effort` &middot; `intermediate`

**How.** Identify the rear leg as the ankle further from the net during preparation. Loading: minimum rear-knee angle in the 500 ms before contact (SIDE_ON only) and mid-hip vertical drop over the same window in metres. Hip drive: net-ward court-plane hip velocity in the 200 ms before contact in m/s, and whether its peak precedes the peak angular speed of the upper arm - a boolean, which is the honest 2D version of the proximal-to-distal idea. Step-back detection: did the rear ankle move away from the net by more than 0.25 m during preparation.

**Bands.** No published band exists for the amateur case; magnitudes are needs user baseline. The one non-baseline output is the boolean 'hip peaked before arm', whose target is true. Directional supporting evidence, labelled: elite forehand smash shows trunk tilt backward 27 +/- 2.8 deg at maximal shoulder external rotation, and elite men hold the racket arm further back with a more extended elbow at the start of the backswing.

**UI.** A single 'engine' panel per overhead grouping the three quantities, because they are one coaching idea, with the boolean as the headline and the metres and degrees underneath. Cue attached: BWF's 'drive up and forwards off the rear leg'.

### 21. Direction changes with an exposed, documented threshold

`medium` &middot; `small effort` &middot; `intermediate`

**How.** footwork_metrics already counts heading turns above 90 deg while moving faster than 1.0 m/s. Two fixes. First, add the speed-dip requirement - speed must fall below and then rise back above a threshold - so slow drift is not counted. Second, and more important, surface the threshold in the editable reference bands and attach a permanent note that this count is definition-dependent and is not comparable to IMU-derived published counts. Tag each change as racket-side or non-racket-side from the turn sign.

**Bands.** IMU reference, printed only with an explicit incomparability warning: 62.4 +/- 22.3 CODs per set in highly trained men's singles (left-side 45.8 +/- 24.1, right-side 16.6 +/- 20.3), where set winners performed significantly FEWER CODs than losers (ES -1.41) - so fewer is better, which is worth saying. Mikiri's own count: needs user baseline, at a fixed threshold held constant across sessions.

**UI.** The count tile carries the threshold in its subtitle ('turns over 90 deg above 1.0 m/s') and a footnote explaining why it cannot be compared to numbers from wearables. The racket-side split feeds the asymmetry panel.

### 22. 'What Mikiri cannot see' panel

`must-have` &middot; `small effort` &middot; `all`

**How.** A short, permanent, non-dismissable panel in the UI and a matching page in the export pack listing exactly what the pipeline cannot observe, each with one line of reason: grip type and grip pressure (no finger keypoints, no racket detection), racket-face angle (needs 3D), forearm pronation/supination (rotation about the forearm's own axis, invisible in 2D), shot type - smash vs clear vs drop (shares overhead kinematics; not separable from pose), shot direction straight vs cross-court, shuttle trajectory and speed, whether the shot went in, and split-step timing against an opponent's contact when filming solo. Grip in particular must be named, because it is the most-taught fault in the governing-body syllabi and a user will assume the tool checked it.

**Bands.** Not applicable - this feature exists to prevent bands being inferred where none can exist. Context worth printing: grip is the entire content of Shuttle Time Lesson 1 and half of Lesson 2.

**UI.** A fixed panel at the bottom of the findings view, always visible, never collapsed by default, and reproduced verbatim on page 2 of the export pack. No composite score anywhere in the app is allowed to span measured and unmeasured territory.

### 23. Racket-side / non-racket-side asymmetry index across metrics

`nice-to-have` &middot; `small effort` &middot; `advanced`

**How.** Every per-rep metric already carries a side label once the lunge pack and handedness detection land. Compute a signed index (racket_side - non_racket_side) / (0.5 * (racket_side + non_racket_side)) per metric with a confidence interval from the per-rep spread, rank by magnitude, and surface only the top two with rep counts on each side shown. Always report in racket-side terms, never left/right, so the BWF pattern tables and the coaching literature - which are written for right-handers - mirror correctly.

**Bands.** Product heuristic: flag when |index| > 0.15 AND both sides have at least 6 reps. No numeric asymmetry threshold exists in the badminton coaching literature; BWF's own functional movement screening criteria are qualitative and bilateral. Otherwise needs user baseline.

**UI.** A single two-row panel at the bottom of the detail view, each row a horizontal diverging bar with the rep counts printed at both ends. Suppressed entirely when either side is under 6 reps.

## What to deliberately not build

This list matters as much as the one above. Each entry is something a
reasonable person would ask for, and each would make the app less trustworthy.

- Split-step timing against the opponent's racket contact. This is the metric the coaching literature actually defines, and we cannot have it: the solo use case has no opponent in frame, and even when one is present they are further from the camera, their keypoints are noisier, and our contact detector's own latency would be added to theirs. A mis-detected opponent contact produces a SYSTEMATIC error, so the player would drill toward a wrong target - worse than reporting nothing. Ship the self-contained reductions instead: presence rate, split-to-first-step latency, stance width, distance from base.

- Recovery-pattern appropriateness after a rearcourt shot (BWF Level 2's differentiated recovery table). Same dependency: it needs the opponent's reply timing as the context variable. Revisit only if opponent tracking (Phase 1b) ever lands and proves reliable.

- Shot type classification - smash vs clear vs drop. Established as out of reach from pose: they share overhead kinematics and differ in racket face and post-impact shuttle behaviour. A racket-mounted IMU study still lost eleven points on the distinction. Keep strokes.py's honest two-tier design (descriptor always, shot name never asserted).

- Shot direction, straight vs cross-court. Needs the shuttle. Not available and not approximable - a guess here would poison the contact-geometry frame, which currently uses 'toward far-court centre' as an explicit approximation rather than a claim.

- Grip type, grip pressure, racket-face angle, forearm pronation/supination. No finger keypoints, no racket detection, and pronation is a rotation about the forearm's own axis that is invisible in 2D regardless of resolution. These are the most-taught faults in the syllabi, which is exactly why they belong in the 'cannot see' panel rather than in a metric that approximates them.

- Proximal-to-distal sequencing score for the overhead. Tempting because BWF teaches the principle explicitly, but the distal segments' peaks are separated by a few frames at 60 fps and the rotation carrying most of the racket speed is invisible in 2D. Anything we computed would be a plausible-looking ordering of the wrong segments. Cut - not even as a detail-view timeline, because a timeline implies a measurement.

- A learned shuttle detector (TrackNet-style). Already established as too slow on CPU with licence problems. The hit-anchored classical local search in the existing roadmap is the only shuttle work worth attempting, and nothing in this roadmap depends on it.

- The racket detector refinement (RTMDet / COCO 'tennis racket'). Already tested and parked at about a third of frames detected. Leave parked; the forearm-extension estimate with a visible caveat is more honest than an intermittently better number that silently changes accuracy between reps.

- Any composite 'stroke quality score' or overall technique rating. It would necessarily span measured and unmeasured territory - the app cannot know whether the shot went in, went where it was aimed, or won the rally. A single tidy number is the single fastest way to lose the credibility that every honest metric depends on.

- Knee valgus / frontal-plane projection angle presented as an injury risk score. FPPA predicts frontal-plane knee moments only weakly to moderately (R-squared 0.12-0.45, essentially zero for step-down) with no established clinical cut-off, and badminton smash-landing knee abduction is only 1.2-2.6 degrees. Build the knee-over-foot ALIGNMENT trend if the rear view is available, as a movement cue with no red styling and a plain 'this is not a medical assessment' line - never a risk number.

- Movement-component classification against the BWF pattern tables (chasse vs cross-behind vs running step). Genuinely valuable and genuinely documented, but step segmentation from 2D pose at consumer frame rates is fiddly and cross-behind detection needs to resolve which ankle is nearer the camera. It would over-flag legitimate stylistic variation, which BWF itself warns against. Revisit after the split-step and lunge detectors are validated against hand-labelled footage.

- Shots per rally and two-player shot frequency as headline numbers. They are the best-documented figures in the literature and they are two-player counts that a solo clip cannot produce. Report own-contact frequency with the axis labelled as such, and do not silently double it.

- Elite values presented as targets. Elite contact height 2.90 m and jump height 53.6 cm are driven by stature and jumping, not technique quality, and elite women's kick-through would be scored as a fault against elite men's two-footed jump. Elite numbers appear only as labelled context; the primary comparator is always the player's own previous session.

- Real-time or live-camera feedback. Every metric here needs calibration clicks, whole-clip bout segmentation, and distribution over reps. A live mode would force single-rep, uncalibrated numbers - precisely the failure mode the rest of the design exists to prevent.

- Any cloud, account, sharing or coach-review feature. The product is local, offline, and for self-coaching with no coach in the loop. Adding a share target changes what the numbers are for, and every honesty decision above assumes the only reader is the person who filmed it.

- Two-decimal-place output anywhere. Round to what can be defended: contact height to 5 cm, angles to 5 degrees, timings to 20 ms at 60 fps, distances to 0.1 m. Precision theatre destroys trust the first time a number jumps between two near-identical reps.

## Rules for presenting any of it

- Hard cap of three headline findings, always. BWF Level 2 instructs coaches to focus on 2-3 crucial points; the UKCC Level 2 worked example compresses 14 observed faults into 4 goals; the self-video literature finds unstructured video increases cognitive load and undermines learning. Everything beyond three sits behind one 'more detail' control, and the complete metric set appears only in the export pack.

- One attentional cue per viewing, one 10-20 second clip per finding, at most two replays offered. Anchor each finding to both a worst-case rep and a representative-median rep so the user sees the pattern, not the outlier.

- Every cue is an effect cue sourced from the coaching manuals, not a body-part instruction. 'Imagine the ground is hot' for short ground contact; 'balance a shuttle on your head' for an upright lunge trunk; 'brush your hair' for the smash backswing. The number lives in the detail view; the headline is the outcome.

- Print the rep count next to every aggregate, without exception. 'Median 0.61 m over 11 reps' is honest; '0.61 m' is not. Below 8 stroke reps or 15 footwork reps, render no aggregate at all - show individual reps and say why.

- Show spread, not just centre. Median plus IQR plus a strip plot of individual reps with click-to-scrub, so the user can inspect their own outliers rather than trust a median.

- Round to defensible resolution: contact height to 5 cm, angles to 5 degrees, timings to 20 ms at 60 fps, distances to 0.1 m, percentages to whole numbers. No two-decimal output anywhere in the UI.

- Label every band with its provenance in the UI text, using three fixed labels: 'elite reference' (published, context only, never a target), 'product heuristic' (our threshold, arguable, editable), 'your baseline' (the player's own earlier sessions). A band with no label is a bug.

- Label lab-derived reference values as 3D. A knee angle of 78 degrees from a side-on phone is not the 81.7 +/- 8.6 degrees a motion-capture lab measures; state that projection error of 5-15 degrees is expected and show a band rather than a point value.

- Never show a metric its view cannot support. Gated metrics are absent with a one-line reason and a camera-placement diagram, not greyed out with a number visible behind them.

- Propagate the position error ellipse into every derived quantity and draw the far court as the smear it is. Distance carries its +/-; the heatmap kernel is the ellipse, not a fixed bandwidth.

- Report every side in racket-side / non-racket-side terms, never left/right, after detecting handedness once per clip. Mirror the coaching pattern tables accordingly. Left/right labelling silently inverts every asymmetry finding for a left-hander.

- Show the cause, suppress the symptom. When the dependency graph fires, write the causal sentence ('you arrive still facing the net, which is why contact is square-on') and drop the symptom from the list rather than spending two of three slots on one problem.

- When a metric drifts, state which interpretation applies. If depth and speed fall together it is conditioning; if depth falls and speed holds it is technique. A drift finding without an interpretation sends the player to the wrong drill.

- Phrase deviations from taught patterns as observations, not faults, and let the user permanently suppress any flag with 'this is how I play'. The manuals themselves say movement patterns are personal and situational.

- Never use red or danger styling for alignment or landing metrics, and carry a plain line that this is not a medical assessment. The evidence does not support an injury claim and an anxious amateur will read one into a red number.

- Refuse rather than degrade below 50 fps for any timing, split-step or sequencing metric, and say why. Silent tracking failure is the worst outcome: RTMPose emits confident-looking keypoints on motion-blurred smash arms, and every downstream number inherits the error invisibly.

- Keep the 'What Mikiri cannot see' panel permanently visible and reproduce it in the export pack. Saying nothing about grip invites the user to assume it was checked and found fine.

- Make session-over-session change the default view from the second session onward, with a change smaller than the metric's own within-session IQR drawn as 'no change'. This is the defence against novelty decay and against reporting noise as progress.

- Never mix measurement scales in one comparison. IMU-derived distance, direction-change counts and speeds are not comparable to court-homography video figures; when a published number comes from a different instrument, print the incomparability warning next to it or do not print the number.

## What each level should see

### Beginner

Load a clip, click four court corners and the two net-post tops once, click yourself. The app then shows a view badge ("Side-on view - 14 of 21 metrics available") and, on a five-minute shadow-footwork protocol run, exactly three finding cards and nothing else.

The three cards a beginner should see are drawn from a beginner-only pool: split-step presence ("Split step on 9 of 14 direction changes"), lunge presence and depth consistency, and base recovery ("You were still moving when the next movement started on 4 of 19 shots"). Each card carries one plain sentence, one number with its rep count, a 10-20 second clip anchored to one representative rep, and one BWF effect cue phrased as an outcome - "imagine the ground is hot", "balance a shuttle on your head" - never a body-part instruction like "rotate your trunk 12 degrees more".

Counts and percentages, not degrees. Presence and consistency, not magnitude. The court heatmap is shown because it is the app's most trustworthy signal and it names a corner rather than a joint: "You spend 41 % more time in the rear forehand corner." Joint angles are not shown at all at this level; they sit behind "more detail" and in the export pack.

Below the three cards, permanently visible, is the "What Mikiri cannot see" panel, so the beginner never assumes their grip was checked and found fine.

The closing screen is the protocol card for next time, so the second session is comparable to the first.

### Intermediate

Same three-card cap, but the pool opens up and the numbers become magnitudes with bands attached.

Findings are now drawn from: split-step latency and stance width against the elite 0.24 s / 0.51 body-height anchors, lunge depth in metres and the straight-leg-phase rate, contact height as a percentage of the player's own standing reach (95-100 % is a genuinely extended arm), shoulder azimuth at contact (side-on above 60 degrees, square-on below 30), backpedalling percentage and time-to-turn, landing width after the kick-through, per-bout distance and metres-per-minute against the 44.6 m/min match band and the 65-76 m/min drill band, and work:rest against 0.4.

The causal precedence graph matters most at this level, because this is where components exist but are badly linked. When backpedalling and square-on contact both fire, the app shows one card - "You arrive at the rear court still facing the net, which is why contact is square-on" - and suppresses the symptom. When lunge depth and speed both drift down across the session, it says conditioning, not technique.

Every tile shows median, IQR and rep count on its face, with a strip plot of individual reps below it, each dot clickable. Aggregates under 8 stroke reps or 15 footwork reps do not render as a number at all.

The intermediate player also gets the detail view: the full lunge table, the per-overhead contact geometry cards, and the editable reference bands, because at this level the bands are worth arguing with.

### Advanced

Single reps are nearly useless here, so the default view is distributional and comparative.

On opening a second or later session the landing screen is "Since last time": at most five rows, each a metric with its delta and an uncertainty band derived from the per-rep IQR and the position error ellipse, so a change smaller than the noise is drawn as no change. Absolute values are one click away.

The advanced pool is: rep-to-rep consistency scores rather than medians, racket-side / non-racket-side asymmetry indices ranked by magnitude with rep counts on both sides, within-session fatigue drift in terciles with the conditioning-versus-technique interpretation stated explicitly, speed variability against bout length with last session ghosted behind, jump rate against the 0.9/min match figure and the 6.7-8.8/min drill figure, and the court centroid and coverage-shape numbers rather than the picture.

The three-card cap still holds. What changes is that the cards now typically say things like "your rear-corner coverage is 0.22 asymmetric across 11 and 8 reps" or "lunge depth fell 6 cm across the session while speed held, which is technique rather than fatigue" - findings that are invisible in any single rep and impossible for a solo player to see without a camera and a court model.

Jump height and contact clearance in metres are available here but deliberately deprioritised, with the elite jump rate (0.9/min) shown alongside precisely to discourage chasing jump height as a headline number.
