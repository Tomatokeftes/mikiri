# Annotated bibliography

Every entry gives the paper, what it measured and how, the numbers Mikiri
might quote, and one paragraph on what it means here. Numbers are copied
from the papers' own tables or abstracts; where a figure was read off a plot
or a summary rather than a table it says so. Entries are grouped by the
module they bear on. Papers that were located but could not be read in full
are listed at the end with what is known about them.

Conventions: "MS/WS" men's/women's singles, "MD/WD/XD" the doubles events,
"GRF" ground reaction force, "IC" initial contact, "CV" coefficient of
variation.

---

## A. Split step, reaction and first step (`footwork.py`)

### A1. Characteristics of split-step skills of the world's top athletes in badminton

PLOS One, 2024/2025. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0316632

**Method.** BWF World Championships 2023 broadcast video, fixed camera, bird's
eye view, 30 fps. Four court-line reference points, homography to metres.
Validated on 38 court positions: 44.8 +/- 22.6 mm mean error, inside the
40 mm line width. Only the near half of the court was analysed, because the
far half's transformation error was too large. Foot position is the thenar
(ball of the foot), not the ankle. Split-step *onset* is defined as the
moment the opponent hits; *reaction time* runs from that moment until one
foot of the player leaves the ground. 148 split steps from three of the four
top-ranked men's singles players (a fourth was dropped for having only 15).

**Numbers.**

| Player | Reaction time (s) | Stance width / height | Split position from base (cm) |
| --- | --- | --- | --- |
| Vitidsarn | 0.24 +/- 0.06 | 0.52 +/- 0.06 | 55.0 +/- 34.3 |
| Naraoka | 0.24 +/- 0.03 | 0.51 +/- 0.06 | 42.1 +/- 23.9 |
| Prannoy | 0.25 +/- 0.04 | 0.52 +/- 0.11 | 54.5 +/- 33.5 |

No significant difference between players on any of the three.

**For Mikiri.** This is the source of the roadmap's 0.24 s, 0.51 body-height
and 42 to 55 cm anchors, and it is worth knowing what each actually is. The
"reaction time" is *opponent contact to first foot lift*, which is a
first-step launch latency, not the timing of the split-step landing. It is
therefore the number Mikiri's `split-to-first-step latency` becomes once the
opponent's contact is confirmed: latency measured from a confirmed contact
rather than from the hop landing. The instrument is the same as Mikiri's (30
fps, homography, near half only), so the comparison is fair, with one
difference in Mikiri's favour: they read the ball of the foot with no
height correction, Mikiri corrects the ankle bias from the net posts.

### A2. Indicator analysis of the footwork in the world's top badminton athletes based on markerless motion measurement

2025/2026, open access via PMC13083079.

**Method.** Same group, same instrument extended. YOLOv11 person detector,
highest-confidence box, bottom edge of the box taken as the projected centre
of mass on court, homography from four corners, 1920x1080 at 30 fps, no
lens correction. 16 top-30 players from the All England 2024
quarter-finals (8 MS, 8 WS matches) against 8 Japanese high-school players
(top four of each sex). Rally clips selected by hand.

**Indicators.** Speed from consecutive court positions; CV = std(speed) /
mean(speed) as a measure of acceleration and deceleration frequency
independent of absolute speed; total distance per rally; occupancy by kernel
density, split by rally won or lost.

**Numbers.** Top men 2.2 to 2.6 m/s mean in-rally speed across rally lengths.
Men's CV about 1.0 to 1.2 in 1-to-4-shot rallies for both groups; top men
hold about 1.0 into 21-to-50-shot rallies while high-school men fall below
1.0 by 21 to 30 shots. Top women's CV decreases progressively with rally
length; high-school women stay high. Distance per rally by match phase
(early / middle / late): top men 25.0 / 18.8 / 16.3 m, high-school men
22.7 / 17.3 / 20.0 m; top women 21.1 / 21.1 / 22.1 m, high-school women
26.5 / 27.1 / 23.2 m. Occupancy: top players of both sexes symmetric,
roughly circular, centred; high-school players about 1 m nearer the net,
and rotated about 45 degrees in rallies they lost.

**For Mikiri.** Source of the roadmap's item 17 (speed variability) and of the
coverage-shape signatures in `docs/footwork.md`. Two things to carry over.
The bounding-box bottom edge is a worse position estimate than an
ankle-bias-corrected mid-ankle point, so Mikiri may state that its instrument
is at least as good as the one behind the reference. And the CV finding is
sex-specific in direction (men hold it, women reduce it), so the item 17
copy must not present one direction as the elite pattern.

### A3. Biomechanical effects of the badminton split-step on forecourt lunging footwork

Bioengineering 11(5):501, 2024. https://doi.org/10.3390/bioengineering11050501

**Method.** 18 male university players, 7+ years, right-handed. Vicon 8
cameras at 200 Hz, AMTI plates at 1000 Hz. Forehand and backhand forecourt
lunges with and without a preceding split step. Stance from 20 N threshold to
toe-off.

**Numbers.** Forehand lunge with a split step: first vertical GRF peak loading
rate 40.06 vs 33.30 N/kg per % stance (p = 0.04); time between the first and
second GRF peaks 7.5% vs 9.9% of stance (p = 0.03); greater hip abduction
and adduction and *less* knee flexion at contact. Backhand: greater knee
rotation, less ankle rotation. No difference in the overall vertical GRF
waveform.

**For Mikiri.** The split step's downstream signature is in the *lunge that
follows*: a quicker plant and a shorter gap between braking peaks. Mikiri
cannot see GRF, but it can see time from first step to lunge plant, which
gives a consistency check on the split-step detector: lunges preceded by a
detected hop should reach the plant faster than lunges that were not. Also a
caution for the lunge pack: a shallower knee at contact after a split step
is the *expected* pattern, not a fault.

### A4. A split-step shortens the time to perform a choice reaction step-and-reach movement in a simulated tennis task

Journal of Sports Sciences 27(12), 2009 (Uzu, Shinya, Oda).

**Numbers.** Target reached in 764 +/- 106 ms with a split step against
868 +/- 63 ms without (p < 0.001), on a choice-reaction step-and-reach.

**For Mikiri.** The cleanest laboratory number for what the split step buys:
about 100 ms on a reach. Usable as context text on the presence-rate card.

### A5. Split-step timing of professional and junior tennis players

Journal of Sports Science and Medicine, 2017 (Filipcic and colleagues).
PubMed 28210342.

**Method.** Fixed cameras at tournament matches; for every stroke, the time
between the opponent's impact and the player's split step.

**Findings.** Timing differs by game situation, shortest on the serve and
increasing through return and baseline play; players respond faster to first
serves than second; each player has an individual timing they adapt to the
situation. The abstract does not give millisecond values.

### A6. Relationship between split-step timing and leg stiffness in world-class tennis players when returning fast serves

2019, located but not read in full. The abstract available in search
summaries reports world-class returners initiate the split step slightly
*after* the ball strike and use a shorter split duration to synchronise with
the server's action, and argues timing and neuromuscular efficiency of the
split are trainable.

**For Mikiri.** Together with A1 and A5: timing relative to the opponent's
contact is the defined metric in both sports, it is individual, and a
landing *slightly after* the opponent's contact is the elite pattern.
Mikiri's refusal to compute timing against `auto` contacts stands; against
`confirmed` contacts the metric is the standard one.

### A7. Automating video-based two-dimensional motion analysis in sport? Implications for gait event detection, pose estimation, and performance parameter analysis

Scandinavian Journal of Medicine and Science in Sports, 2024 (Mundt and
colleagues). https://doi.org/10.1111/sms.14693

**Findings.** Comparing 2D pose keypoints against projected marker
trajectories: a minimum of 100 Hz is needed to detect key events; a 20 ms
error in touchdown produced knee-angle differences of up to 20 degrees;
marker-based event algorithms transfer poorly to 2D video and new event
detectors are needed.

**For Mikiri.** Direct support for two existing rules: refuse timing metrics
below 50 fps, and round timings to 20 ms. It also says that any joint angle
reported *at an event* (knee at landing, knee at plant) inherits the event
timing error, so those angles should carry a wider band than angles at a
kinematic extreme.

---

## B. Contacts, serves, rallies and the chain (`strokes.py`, `match.py`, `segments.py`, `rally.py`)

### B1. MonoTrack: shuttle trajectory reconstruction from monocular badminton video

CVPR Workshops 2022 (Liu and Wang). https://arxiv.org/abs/2204.01899

**Method.** Court corners plus net-post tips give the projection matrix. A
compact U-Net (2.9 M parameters against TrackNetV2's 11.3 M) detects the
shuttle in 2D. HitNet, a two-layer GRU over 12-frame windows of court
corners, both players' poses and the shuttle's 2D position, classifies each
frame as no hit, near hit or far hit. A dynamic programme over the per-frame
scores then imposes three constraints: hits alternate between players, hits
are at least 0.5 s apart, a rally runs at about one hit per second. Between
hits, the 3D trajectory is a particle under quadratic drag,
d2x/dt2 = g - Cd |v| v, with initial position and velocity solved by
minimising reprojection error plus priors: launch near the hitter (from
pose), end near the receiver, landing inside the court.

**Numbers.** 2D detection 88.6%. Hit detection 76.4% for HitNet alone, 89.7%
(precision 94.9%) with the constrained decode. 3D error on synthetic data
8.0 cm with priors, 14.9 cm without; error falls with flight time. On real
video, reprojection error 8 to 10 px with ground-truth hits and 23 to 37 px
end to end, because a single wrong hit corrupts the shots on both sides of
it; about 20% of shots are affected. Front-court shots and shots leaving the
frame are worst. Singles only, broadcast view only.

**For Mikiri.** Three transfers. (1) The constrained decode is the single
biggest gain in the paper and it needs no model: Mikiri has the swing signal
per player and the same three constraints, and currently applies them
*after* thresholding rather than *during* decoding. (2) The physics model is
the one in `shuttle.py`, and the priors it uses (launch at the hitter's
racket, end at the receiver, land in court) are all things Mikiri knows
from tracking, so a 3D fit from two contacts and a flight time is feasible
with no shuttle detection at all. (3) The failure mode is the one Mikiri's
provenance labels exist for: a contact error poisons two shots, so a shot
derived from an `auto` contact must stay `auto`.

### B2. Towards structured analysis of broadcast badminton videos

WACV 2018 (Ghosh, Singh, Jawahar). https://arxiv.org/abs/1712.08714

**Method and numbers.** Point segmentation by HOG features every tenth frame
and a chi-squared SVM, Kalman-smoothed: F1 95.4%. Players detected as two
classes, top and bottom, by a fine-tuned Faster R-CNN. Player identity across
court switches by colour histograms after background subtraction, clustered
with a Gaussian mixture: 98.0%. The serve is not detected as an action;
instead the server is "usually closer to the mid-line than the receiver", a
polynomial SVM on the first 50 frames of the two tracks picks the server, and
the previous point's winner is read off it: 94.1% outcome accuracy. Stroke
segmentation by temporal convolutional networks over box crops: 80.5% edit
score.

**For Mikiri.** Prior art for "the winner is whoever serves next", eight
years before `match.py`, with an accuracy figure for it. The serve cue worth
stealing is positional: at rally start the server stands nearer the centre
line and both players stand still, which catches a low serve whose swing the
detector misses. And the colour-histogram identity is the doubles partner
discriminator (section E).

### B3. Towards real-time analysis of broadcast badminton videos

2023. https://arxiv.org/abs/2308.12199

**Numbers.** Rally frames by a fine-tuned ResNet-18: F1 97.2%. Rallies are
runs of rally frames with gaps of at least 3 s between them and a minimum
length of 3 s. YOLOv3 player boxes at 98% mAP. Distance from homographed box
positions had a mean error of 4.5 to 4.7 m per rally against ground truth on
25 rallies, with deviations of 1 to 6 m.

**For Mikiri.** A cautionary comparator for distance: box-based court
position costs metres per rally. Mikiri's ankle track with propagated error
is the better instrument and the export can say why. The 3 s rally gap is
their version of `MERGE_GAP_S`.

### B4. Automated hit-frame detection for badminton match analysis

2023. https://arxiv.org/abs/2307.16000

**Method.** Broadcast MS video at 30 fps. A small CNN classifies frames as
high-angle (rally) or other: 99.8%, giving rally trimming at 80.6% (the
misses are broadcast camera changes). A court R-CNN finds six court
keypoints, used to keep only on-court people. A transformer over both
players' 17-keypoint sequences predicts, per frame, whether the shuttle is
steady, flying to the bottom court or flying to the top court (over 92%
accuracy per token). A hit frame is a token change.

**Numbers.** Hit frames within +/-5 frames: F1 63.9%; within +/-15 frames:
F1 79.6%; within +/-25: 81.1%.

**For Mikiri.** The interesting part is the representation: the shuttle's
*direction of travel* is predicted from both players' poses, i.e. the
receiver's behaviour carries information about the hitter's contact. That
is the receiver-reaction corroboration idea in `ideas.md`. The frame
tolerances also calibrate expectations: pose-only hit timing is good to
about half a second, not a frame.

### B5. A new perspective for shuttlecock hitting event detection

2023, CoachAI challenge entry. https://arxiv.org/abs/2306.10293

MobileNetV2 plus BiLSTM over background-subtracted optical flow, with
YOLOv8-pose and TrackNetV2 inputs. Scored to +/-2 frames in the challenge
and did poorly at that tolerance. Included because it confirms that
frame-accurate hit timing from appearance alone is hard even with a learned
model, and that the shuttle track is what makes it easy.

### B6. An all-deep system for badminton game analysis

2023, CoachAI challenge. https://arxiv.org/abs/2308.12645

YOLOv7 for shuttle, players, court, net and racket; an asymmetric U-Net
variant of TrackNet (F1 0.94 at 640 px, 188 GFLOPs); X3D and EfficientNet
heads for hit frames, ball height, landing position, player locations,
backhand, round-the-head, ball type and winner. Final challenge score 0.78.
The authors note that detection that "seems satisfactory" is not accurate
enough for exact-location tasks and that shuttle position error propagates
into every downstream label. This is the error-compounding argument for
Mikiri's provenance labels, made by people who built the pipeline without
them.

### B7. Service and end-of-rally detection in badminton videos; Ready-to-serve detection in badminton videos

Springer 2024 and IEEE 2024. Abstracts only. Service detected from court
details and poses: twelve frames by 26 pose-derived features (distances,
angles, gradients, movements typical of serving and receiving) arranged as a
2D array for a CNN, or a BiLSTM; end of rally from a gradient of activity.
No accuracy figures were readable. Included as evidence that pose-derived
serve detection is a standard, small-feature problem, and that the
receiver's posture is part of the signal.

### B8. Detecting scoreboard updates to increase the accuracy of ML automatic extraction of highlights in badminton games

Springer 2023. Abstract restricted. A YOLOv5 model reads the broadcast
scoreboard, split into the two teams' halves, and score changes anchor
highlights. Also: ScoreSight, an open-source scoreboard OCR tool
(https://github.com/royshil/scoresight). Relevant to Mikiri only as *test
tooling*: on broadcast benchmark clips, the on-screen score is a free oracle
for the rally chain.

---

## C. Shot naming and the shuttle (`shots.py`, `shuttle.py`)

### C1. ShuttleSet: a human-annotated stroke-level singles dataset for badminton tactical analysis

KDD 2023. https://arxiv.org/abs/2306.04948 and https://github.com/wywyWang/CoachAI-Projects

44 matches, 104 sets, 3,685 rallies, 36,492 strokes, 27 top players, 2018 to
2021. Eighteen shot types, the hitting location, and both players' locations
at every stroke. The arXiv page carries CC BY 4.0; `docs/tactical.md` says
MIT. **Check the repository licence before deriving shipped bands from it,
and record which licence applies.**

### C2. BST: Badminton stroke-type transformer for skeleton-based action recognition in racket sports

2025. https://arxiv.org/abs/2502.21085

**Method.** Inputs: 2D poses from RTMPose (the authors tried 3D estimates and
found 2D better), shuttle 2D positions from TrackNetV3, court positions from
line detection. ShuttleSet reduced to 33,481 strokes in 40 matches and 25
merged classes; BadmintonDB (two players, 7,658 strokes, 18 classes);
TenniSet (3,569 strokes, 6 classes).

**Numbers.** 82.5% on ShuttleSet's 25 classes. Ablation: shuttle trajectory
alone (81.0%) beats player positions alone (79.4%). Confusions are between
semantically adjacent classes: smash against wrist smash, return net against
defensive return drive. Depends on hit-frame detection and shuttle tracking
being right.

**For Mikiri.** The shuttle-beats-pose result the docs already cite, now with
the modern number, and from a pipeline that uses the same pose model as
Mikiri. The confusions are within the smash family and within the net
family, which is what a per-pair confusion report would surface.

### C3. TemPose-TF-ASF: two-stage bidirectional stroke context fusion for badminton stroke classification

2026. https://arxiv.org/abs/2605.02558

A first pass classifies each stroke; a second pass feeds the *preceding and
following* strokes' predictions back in as context. Consistent gains in
accuracy and macro-F1 over the baseline; the abstract does not give numbers
and does not address player-specific adaptation.

**For Mikiri.** The neighbouring strokes are the strongest context, and in a
rally the neighbours are the opponent's strokes. Mikiri does not need a
second network to use that: the reply's timing, zone and contact height are
the context, and they come from tracking.

### C4. Widely applicable strong baseline for sports ball detection and tracking (WASB)

NTT, 2023. https://arxiv.org/abs/2311.05237, code MIT at
https://github.com/nttcom/WASB-SBDT

HRNet-style high-resolution features, N frames in and N heatmaps out. On the
TrackNetV2 badminton data, F1 93.1% at 34 fps or 91.6% at 70 fps on a V100,
against TrackNetV2's 90.5% at 77 fps. Tracking is a local motion model
p_next = p + v + a/2 with candidate gating, chosen over Kalman and particle
filters. Datasets for soccer, tennis, badminton, volleyball and basketball.

**For Mikiri.** Independent confirmation of the three-point
constant-acceleration extrapolator in `shuttle.py`. GPU figures only.

### C5. TrackNetV3: enhancing shuttlecock tracking with augmentations and trajectory rectification

ACM Multimedia Asia 2023. Code MIT at https://github.com/qaz812345/TrackNetV3

Background estimate as an auxiliary input, mixup augmentation, a
rectification module for occluded frames. 97.5% accuracy on its own
Shuttlecock Trajectory Dataset, 25 fps in their environment. A TensorRT INT8
fork exists. The weights' licence is not stated in the readme summary read
here; the code is MIT.

**For Mikiri.** `docs/shuttle.md` records "the available checkpoint's licence
is unresolved". The code licence is MIT; the weights need a separate check.
The other objection, CPU cost, is answered in `ideas.md` item 6 by running a
small net only in the hit-anchored crop.

### C6. Enhancing badminton game analysis: an approach to shot refinement via a fusion of shuttlecock tracking and hit detection from monocular camera

Sensors 24(13):4372, 2024. https://doi.org/10.3390/s24134372

Not readable in full. From the abstract and search summaries: TrackNet for
the shuttle track, YOLOv7 to detect whether a player is swinging, and the two
fused to refine shot boundaries and shot types from a single camera without
wearables. Same architecture of ideas as `shots.py` plus `strokes.py`.

### C7. A real-time, vision-based system for badminton smash speed estimation on mobile devices

2025. https://arxiv.org/abs/2509.05334

YOLOv5 shuttle detection, Kalman tracking, court calibration for scale,
drag-aware speed. The limitations section lists the ones Mikiri already
states: single-camera depth ambiguity, 30 fps, rolling shutter, lighting.

### C8. The physics of badminton

New Journal of Physics 17, 2015 (Cohen, Texier, Quere, Clanet).
https://doi.org/10.1088/1367-2630/17/6/063001

Quadratic drag; aerodynamic length of a few metres; the trajectory bends
toward a vertical asymptote, unlike a parabola; feather and plastic differ
by about 1 m (10%) in range at the same launch. Terminal velocity measured
elsewhere at 6.79 m/s (Aerodynamics of a badminton shuttlecock, wind tunnel).
Source of `DRAG_LENGTH_M = 4.5` in `shuttle.py`.

### C9. Shuttlecock velocity decay after smash and slice shots in badminton

2026. https://arxiv.org/abs/2601.01412

High-speed video of elite smashes: velocity decays exponentially with
distance, halving every 3.35 m for a fast feather shuttle (a "speed index"
sets the exact figure); plastic shuttles deform at speed and their drag
coefficient falls; slice spin slows the shuttle.

**For Mikiri.** Halving every 3.35 m is an e-folding length of 3.35 / ln 2 =
4.8 m, consistent with the 4.5 m already used. It also quantifies the
sentence in `docs/shuttle.md` about speed at impact being unmeasurable at
30 fps: at 100 m/s the shuttle covers 3.3 m per frame, i.e. it loses half
its speed in one frame interval.

### C10. YO-CSA-T: a real-time badminton tracking system utilizing YOLO based on contextual and spatial attention

2025. https://arxiv.org/abs/2501.06472

YOLOv8s plus attention: 90.4% mAP at IoU 0.75, over 130 fps on a GPU.
Included for completeness; GPU-class.

### C11. Left- versus right-handed badminton slice shots: opposite shuttlecock spinning and Magnus effect

2023. https://arxiv.org/abs/2310.11155

Slice by a left-hander spins the shuttle the opposite way; the Magnus effect
curves and slows it. Relevant only as a reason a two-endpoint launch
estimate should carry a wider band for sliced shots, which Mikiri cannot
detect; label it.

---

## D. Biomechanics of the lunge, landing and overhead (`movements.py`, `coaching.py`)

### D1. Impact of gender, lunge direction, and fatigue on the lower limb biomechanics in badminton lunges

PLOS One 2025. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327428

24 amateur players (12 M, 12 F). Fatigue by sprints and maximal jumps to
70% jump height, RPE 17 or 80% HRmax. Stance-phase peak knee flexion
81.7 +/- 8.6 unfatigued, 77.5 +/- 8.3 fatigued (p = 0.028); hip internal
rotation 18.7 to 13.9 degrees; leg stiffness 9.03 to 10.49 BW/m; forehand
ankle inversion 12.0 vs backhand 21.4 degrees; second GRF peak 1.75 BW
(women) vs 1.65 (men). Source of the roadmap's item 9 and 18 numbers.

### D2. Lower limb biomechanics and knee tissue loading under fatigue during the badminton forward lunge in recreational female players

Frontiers in Public Health 2026. 16 recreational women. After fatigue: less
knee flexion in early and mid stance, less dorsiflexion over 0 to 36% of
stance, more eversion over 69 to 93%, more knee varus and valgus moment at
peak GRF; finite-element ACL strain +18.4%. **For Mikiri:** direction only.
The roadmap's refusal to present any of this as injury risk stands; the app
may say knee flexion at the plant tends to fall late in a session, and stop.

### D3. Functional data analysis of lower-limb joint kinematics during badminton lunges under fatigue

PMC12920591, 2026. 17 high-level men, shuttle runs plus jumps to 70%.
Reduced hip and knee sagittal and frontal range, increased ankle dorsiflexion
and inversion-eversion range: "proximal stiffening, distal mobilisation". No
degree values in the results.

### D4. In vivo knee biomechanics during badminton lunges at different distances and foot positions (dual fluoroscopy)

PMC10768190, 2023/2024. Leg length defined as anterior superior iliac spine
to floor. Lunge at 1.5 x leg length versus maximum lunge: braking-phase knee
flexion 57.3 +/- 7.8 vs 61.8 +/- 9.6 degrees; +4.0 at IC, +5.2 at peak GRF.
External foot rotation of 20 degrees changes femoral translation but little
flexion.

**For Mikiri.** A self-normalised lunge-depth anchor: 1.5 x leg length is the
standard experimental lunge, and the maximum is beyond it. Mikiri's lunge
detection threshold of 1.2 x leg length ankle separation sits sensibly below
the standard lunge. Knee flexion at a standard lunge is about 60 degrees at
the braking phase, lower than the 80-degree stance-phase peak in D1; the two
are different instants and the lunge table should say which it reports.

### D5. Biomechanics of lower limb in badminton lunge: a systematic scoping review

PeerJ 8:e10300, 2020. https://peerj.com/articles/10300/

Studies standardise lunge distance by leg length or foot length (for example
2.5 foot lengths forward, 3 backward). Elite versus sub-elite: higher impact
attenuation, a more aggressive knee and ankle moment strategy, higher medial
plantar load, more push-off GRF, less variable footstrike angle, better
eccentric braking and faster recovery. **For Mikiri:** the two elite markers
visible in 2D are footstrike-angle *consistency* and recovery time after the
plant, both of which the lunge pack and the base-recovery item can carry as
"your baseline" metrics.

### D6. Impact of unanticipated and backhand-area smash landing on the lower limb biomechanics of female badminton players

PMC12162505, 2025. 13 national-ranked women. Knee flexion at IC: rear-court
jump smash -24.2 (anticipated) / -23.1 (unanticipated) degrees; lateral jump
smash -18.0 / -16.3. Unanticipated landings: higher instantaneous load rate,
higher knee extension moment at IC, more vertical GRF at 53 to 62% of
landing. Source of the roadmap's "16 to 24 degrees at IC is normal" and of
the note that surprise loads harder than habit.

### D7. Biomechanical determinants influencing smash performance in racket sports 2000 to 2024: systematic review and meta-analysis; and Kinematic secrets of the overhead forehand smash through novice-skilled comparison

2024, Applied Sciences 13(22):12488, 2023. Trunk rotation (X-factor),
shoulder internal rotation and proximal-to-distal sequencing determine
post-impact shuttle speed; skilled players use more trunk rotation than
novices. **For Mikiri:** consistent with item 11 (shoulder azimuth) as a
directional heuristic and with the refusal to score proximal-to-distal
sequencing from 2D.

### D8. Using markerless motion analysis to quantify sex and discipline differences in external mechanical work during badminton match play

Journal of Sports Sciences 43(12), 2025, University of Bath. Eight cameras,
HRNet pose driving an OpenSim model, 14 men and 14 women, 9 singles and 6
doubles matches, 30 points per match. Normalised work and power greater in
men and in singles (effect size 0.60 +/- 0.29). Context for the doubles
work-rate bands: doubles is lower per-player mechanical work despite
higher shot frequency.

### D9. BadmintonGRF: a multimodal dataset and benchmark for markerless ground reaction force estimation in badminton

2026. https://arxiv.org/abs/2605.01876

Eight synchronised RGB views at about 120 fps, four Kistler plates, Vicon;
10 players, 156 instrumented trials, 17,425 impact-segment archives, 12,867
usable view-specific instances. Tier 1 (public, CC BY-NC 4.0): 2D pose
sequences time-aligned with GRF and metadata. Tier 2 (controlled): raw video
and mocap. Benchmark task: 2D pose to GRF.

**For Mikiri.** Force plates give ground-truth IC, toe-off and flight for
every trial, at 120 fps. That is the labelled footage the split-step and
lunge detectors say they lack, in pose form, from the same class of pose
model. Use for a validation test suite (subsampled to 30 and 60 fps to
measure what each frame rate loses), never as shipped data, and check that
the movements included cover split steps and lunges before relying on it.

---

## E. Pose validity, frame rate and camera placement (`geometry.py`, `coaching.assess_trust`)

### E1. Influence of the camera viewing angle on OpenPose validity in motion analysis

PMC11819822, 2024/2025. Front lunges, six repetitions, four cameras at 45
degree intervals (front-left, front-right, back-left, back-right) against
mocap. Mean percentage deviation excluding the shoulder: front-right 15%,
front-left 24%, back-left 13%, back-right 13%. Knee 8 to 27%, hip 11 to 32%,
elbow 5 to 18%, shoulder 18 to 49% from every angle. Recommendation: use the
angle with the fewest occlusions of major keypoints; back views were best.

**For Mikiri.** Adds to the view gate: an oblique view from *behind* is
tolerable for lunge angles, an oblique view from the *front* is the worst
case, and shoulder angles are poor from every single view. The placement
diagram should mark the front-oblique position as the one to avoid.

### E2. Validity of 2D pose for sagittal joint angles (gait and jump studies)

Several studies (PMC11054247, PMC9399401, PMC10635560, the sports validation
study of walking, countermovement jump and throwing). Sagittal errors: hip
4.0, knee 5.6, ankle 7.4 degrees in gait; under 3.2 degrees flexion-
extension in a vertical jump; keypoint position errors mostly under 30 mm,
one in ten over 40 mm.

**For Mikiri.** The 5 to 15 degree band the roadmap attaches to lab
comparisons is the right order; the low end applies to slow sagittal motion,
the high end to fast, oblique or occluded frames.

### E3. AthletePose3D: a benchmark dataset for 3D human pose estimation and kinematic validation in athletic movements

CVPR Workshops 2025. https://arxiv.org/abs/2503.07499

1.3 M frames, 165 k postures, 12 sports motions at high speed. Monocular 3D
models trained on conventional data: 214 mm MPJPE; fine-tuned on athletic
data: 65 mm. Models capture joint angles and individual differences but
"struggle with high-speed velocity metrics".

**For Mikiri.** A 2D-to-3D lifter is a plausible future backend for *angles*
(it would lift the SIDE_ON gate on knee and trunk angles) and an implausible
one for racket speed. Whether badminton is among the 12 motions was not
confirmed.

### E4. Temporal smoothing: SmoothNet (ECCV 2022) and the One Euro filter

SmoothNet learns long-range per-joint temporal refinement on top of any
estimator; the One Euro filter is a first-order low-pass with a speed-
adaptive cutoff (low jitter at rest, low lag in motion). Relevant to
`smoothing.py` if the split-step signal is ever found to be limited by
jitter rather than by resolution.

### E5. Single view metrology from a known height

Forensic height estimation from a vertical vanishing point and one known
reference height is accurate to about 2 cm on surveillance video. **For
Mikiri:** when the net posts are out of frame, a player of known height
standing upright at a known floor point is a vertical reference of the same
kind as a post, and the camera height can be recovered from it with a wider
band; see `ideas.md` item 12.

---

## F. Time-motion and notational reference values (`coaching.py`, `rally.py`, `segments.py`)

### F1. Match-play data according to playing categories in badminton: a systematic review

Frontiers in Sports and Active Living, 2025.
https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1466778/full

| Category | Rally (s) | Rest (s) | Work density | Effective playing time (%) | Shots per rally | Shots per s |
| --- | --- | --- | --- | --- | --- | --- |
| MS | 8.8 +/- 2.0 | 22.9 +/- 10.6 | 0.4 +/- 0.1 | 34.1 +/- 5.3 | 8.9 +/- 2.1 | 1.0 +/- 0.2 |
| WS | 9.3 +/- 2.0 | 17.2 +/- 4.4 | 0.4 +/- 0.1 | 29.0 +/- 6.6 | 6.9 +/- 1.1 | 0.8 +/- 0.2 |
| MD | 6.5 +/- 0.7 | 24.9 +/- 3.4 | 0.2 +/- 0.0 | 19.2 +/- 2.3 | 9.5 +/- 1.3 | 1.5 +/- 0.0 |
| WD | 9.7 +/- 0.9 | 22.8 +/- 4.0 | 0.4 +/- 0.0 | 27.2 +/- 5.0 | 11.8 +/- 1.7 | 1.3 +/- 0.1 |
| XD | 6.8 +/- 1.1 | | | | | |

Shot-type share, MS: net 24.5 +/- 7.5, smash 20.3 +/- 9.4, clear 16.4 +/- 5.3,
drop 12.7 +/- 5.2, drive 9.9 +/- 11.9. WS: net 23.3 +/- 6.8, clear
22.6 +/- 12.3, drop 17.7 +/- 13.9, smash 14.6 +/- 6.3, drive 4.6 +/- 3.7.
The standard deviations on the shot shares are large because the studies
coded shots differently; this is the "small hand-notated samples" caveat in
`docs/tactical.md`, quantified.

**For Mikiri.** Every figure in `feature-roadmap.md` item 2 matches. This is
the citation to put on the band file.

### F2. Notational analysis and physiological and metabolic responses of male junior badminton match play

Sports 11(2):35, 2023. Five matches, 427 points. Rally 5.7 +/- 3.7 s, rest
11.2 +/- 5.9 s, shots per rally 5.6 +/- 3.6, work density 0.50 +/- 0.21,
effective playing time 32.3 +/- 8.4%, shot frequency 1.04 +/- 0.29 per s.
86% of rallies under 10 s, mode 4 to 6 s; 67% of rests under 12 s, 13% over
18 s. Second game longer than the first (rally 5.2 to 6.1 s).

**For Mikiri.** The nearest thing to a club band. Rests are half the elite
figure and mostly under 12 s, which matters for `MIN_REAL_REST_S` (8 s) and
for the segmenter's play/rest split on club footage: a club rest of 8 to
12 s is normal, not evidence of editing.

### F3. Performance differences of temporal parameters and point outcome between elite men's and women's badminton players according to match-related contexts

IJERPH 2019 (Gomez, Rivas, Connor, Leicht). PMC6862575.

Rally 9.9 s (men) vs 9.1 (women); rest 22.9 vs 22.0; strokes per rally
10.6 vs 8.7; in long matches rally 12.1 vs 10.5 s and strokes 12.6 vs 10.1.
Point outcomes when serving: winners 19.1% (men) vs 15.1% (women); forced
errors 13.8 vs 15.5. When receiving: unforced errors 18.1 vs 21.6.

### F4. Changes in the game characteristics of a badminton match: a longitudinal study through the Olympic Game finals analysis in men's singles

Journal of Sports Science and Medicine 2015 (Laffaye, Phomsoupha, Dor).
PMC4541123. Point endings across finals: 36% winners, 23% forced errors,
41% unforced errors (from the search summary; the full text was not
readable). The smash ends a rally by a direct point in 29.1 +/- 8.4% of
cases.

### F5. Comparison of the temporal and technical-tactical characteristics in badminton men's singles under different competition formats

PMC12463982, 2025. Individual versus team events: shots per rally
10.05 +/- 2.21 vs 9.57 +/- 1.61; rest 21.4 +/- 4.5 vs 20.4 +/- 4.6 s; over
69% of rallies in both formats last 0 to 10 s. Point endings: forced errors
19.8% vs 27.0%, unforced 30.0% vs 25.6%. Strokes coded into ten
subcategories over net, mid and back court with direction and landing zone,
inter-rater kappa above 0.85.

**For Mikiri.** The coding scheme (ten stroke subcategories by three zones)
is a reasonable target vocabulary for the review queue's shot names, and the
0.85 kappa is what trained human coders achieve, which bounds what the
per-player classifier's held-out agreement can be expected to reach.

### F6. Entropy of badminton strike positions

PMC8304171, 2021. 259 elite women's matches 2015 to 2019, 218,081 strokes,
63 players, 12 zones (three deep by four across). The four corner zones
hold 17%, 16%, 13.9% and 13% of all strokes; normalised entropy 0.894; no
difference between winning and losing positions.

**For Mikiri.** An elite reference for the contact-zone distribution of the
descriptor: about 60% of elite women's contacts are in the four corners.
Also an argument for a corner-count tile on the strokes panel that needs no
shot name.

### F7. Using inertial measurement units to quantify external load in highly trained men's singles by set outcome

Journal of Sports Science and Medicine 24(4), 2025. 18 highly trained men,
110 sets, Catapult Vector S7 at 100 Hz. Set winners had lower player load,
less distance, fewer explosive efforts, changes of direction and
accelerations than losers overall; but in close sets (0 to 5 point gap)
winners performed *more* explosive efforts and changes of direction, and
fewer only when the gap was 6 or more.

**For Mikiri.** Corrects `feature-roadmap.md` item 21, which says "fewer is
better, which is worth saying". It is worth saying only with the score-gap
qualifier, and Mikiri does not know the score gap on a solo clip, so the
sentence should go.

### F8. Squash: quantifying motion and T dominance of elite squash players; Tactical use of the T area by players of differing standard

Journal of Sport and Human Performance; and a 2009 study. Broadcast video,
inverse perspective mapping, Dartfish tracking. Correlation between world
rank and mean distance from the T: r = 0.64, p < 0.001. Distance error 3.7%
against manual tracking.

**For Mikiri.** The sibling sport's validated position metric is distance
from the central base *at the moment the opponent strikes*. Mikiri's base
metrics measure the same thing against the player's own moves; against
confirmed opponent contacts it becomes the squash-validated quantity.

### F9. Validation of a video-based system for automatic tracking of tennis players

2018. Mean bias 0.33 m, precision 0.15 m, accuracy 0.36 m for player court
position from video. A comparator for Mikiri's 5 cm RMS claim with the net
posts and a few tens of centimetres without.

### F10. Multi-task tennis stroke biomechanics analysis using MediaPipe pose

2026. https://arxiv.org/abs/2606.15992

Stroke detector s(t) = 0.5 v_wrist + 0.3 m_elbow + 0.2 m_shoulder on
metric, hip-centred world coordinates; peaks above a per-player threshold
(0.07 to 0.08 for professionals, 0.04 for the amateur), 40-frame refractory
gap. Using metric coordinates instead of image-plane landmarks raised
cross-player stroke-type accuracy from 47% to 83%. Direction (61.9%) barely
beat the majority class.

**For Mikiri.** Nearly the same swing signal as `strokes.py` (weights 0.5,
0.3, 0.2 and a body-height normalisation), arrived at independently. The
transferable finding is the *per-player threshold*: amateurs swing smaller,
so `MIN_PEAK_SPEED_BH` may be a place where one constant does not fit both
players on a clip. And direction from pose fails in tennis too, which is why
the reply-endpoint approach in `ideas.md` item 1 avoids pose for direction.

---

## G. Doubles, identity and formation (`tracking.py`, future formation pack)

### G1. Bridging the gap: doubles badminton analysis with singles-trained models

2025. https://arxiv.org/abs/2508.13507

YOLOv8 detection, BoT-SORT tracking, ViTPose keypoints for four players;
identity from jersey appearance and court position and formation patterns;
singles-trained stroke models fine-tuned on doubles. Limitations: identical
uniforms within a pair, rapid position swaps, occlusion.

### G2. Estimation of control area in badminton doubles with pose information from top and back view drone videos

Multimedia Tools and Applications, 2023. First annotated drone dataset (top
and back views) of doubles; a network builds a control-area probability map
from a Gaussian mixture of player positions and a graph convolution over
poses; control area correlates with scoring.

**For Mikiri.** The learned map is not needed to get the metric that
matters. A pair's coverage from two positions with error ellipses, the seam
between partners, and the gap area beyond reach of both, are all geometry.

### G3. Enhancing sports strategy with video analytics and data mining: automated video-based analytics framework for tennis doubles

2025. https://arxiv.org/abs/2507.02906

GroundingDINO with a text description per player ("red-shirt player") plus
YOLO-pose; formations labelled from the quadrant each player occupies;
eight videos. **For Mikiri:** the quadrant-tuple encoding of a formation is
a clean representation, and asking the user to describe the partners once
is a reasonable UI for seeding a colour model.

### G4. Colour-histogram re-identification in sports tracking

Ghosh et al. (B2): 98.0% player identity across court switches from colour
histograms and a Gaussian mixture. Team-aware football tracking with SAM
(2025): HSV histograms of the jersey region improve recovery after
occlusion. SportSORT and McByte++ (2025) add jersey number and team ID as
domain features. **For Mikiri:** the cheap floor for partner identity is an
HSV histogram of the torso box per side, with attribution withheld when the
two partners' histograms are not separable.

---

## H. Anticipation, deception and feedback (`coaching.py`, review queue)

### H1. The use of contextual information for anticipation of badminton shots in different expertise levels

2022. PubMed 35040748. Video occlusion test; 26 elite, 15 competitive, 17
novice. Side (direction) was predicted better than length by everyone;
elites beat novices on both; adding rally context slowed responses without
improving accuracy.

**For Mikiri.** Two implications. Humans read direction from the hitter's
body better than they read length, so the roadmap's ordering (type is
learnable from pose, direction is not) is not what human anticipation
suggests; the honest position is that neither is measured from pose, and
both come from the reply. And club players are readable, which is the premise
of the per-player classifier.

### H2. Self-video modeling combined with self-feedback in youth sport: an opinion on cognitive load, attention, and learning design

Frontiers in Psychology, 2026. Concrete rules: clips of 10 to 20 s with a
single task focus; one attentional cue per viewing; at most two viewings;
before/after comparison clips; structured self-feedback prompts (Strength:
a specific action; Improvement: a goal-linked change). The evidence on
*timing* of feedback is noted as insufficient.

### H3. Meta-analysis of the reduced relative feedback frequency effect on motor learning and performance

Psychology of Sport and Exercise, 2022. The guidance hypothesis (feedback
after every trial helps performance but harms learning) is not supported
across the literature. Bandwidth feedback (only when outside a tolerance)
remains well supported for simple skills. Self-controlled feedback has a
separate meta-analytic literature with positive but modest effects.

**For Mikiri.** The three-finding cap should not be justified by the
guidance hypothesis. It is justified by coach practice (BWF, UKCC) and by
cognitive load (H2). Bandwidth feedback is a good description of what the
bands already do: no finding inside the band.

### H4. Self-controlled video feedback facilitates the learning of tactical skills in tennis

Research Quarterly for Exercise and Sport, 2023/2024. Learners who chose
when to view video with attentional cueing learned tactical decisions better
than yoked controls. Supports "everything is editable afterwards" and letting
the user choose which finding to open.

---

## I. Court calibration (`court.py`)

### I1. Robust automatic camera calibration in badminton court recognition (IEEE 2022); A court line extraction algorithm for badminton tournament videos with horizontal line projection learning (IET Image Processing 2023); CourtKeyNet (2026)

Line-candidate extraction, Hough, model fitting with line-count constraints
and a 3D mapping filter; a learned horizontal-line projection; and a
keypoint network with polar attention and geometric constraints. All are
full automation of what Mikiri asks the user to click. The permissive
licence claimed for CourtKeyNet in a search summary was not verified.

**For Mikiri.** Full automation is not the win; snapping is. The benchmark
manifest already records that a least-squares fit of each painted line to
its own pixels reached 0.35 cm reprojection where hand clicks reached 14 cm.
Productising that as "click near the corner, Mikiri snaps to the line
intersection" removes the largest term in the clicked-landmark error with
no model.

---

## Located but not read

* Tracking players in a badminton court by two cameras (2023): CSRT trackers,
  a top view and a 45-degree rear view, HOG plus SVM re-identification. No
  quantitative accuracy.
* Vision-based movement recognition reveals badminton player footwork using
  deep learning and binocular positioning (Heliyon 2022): shoe detection,
  stereo 3D at 74.7% positioning accuracy.
* Stroke recognition in badminton videos based on pose estimation and
  temporal segment networks (Journal of Image and Graphics).
* Time series classification of badminton pose using LSTM with landmark
  tracking (2024): serve, lob and smash at over 90%, small data.
* Video pose distillation for few-shot fine-grained sports action recognition
  (ICCV 2021): pose-derived features give 7 to 23 points on 8 to 64 examples
  per class in diving and gymnastics, the closest published analogue to a
  per-player classifier trained on a few dozen confirmed strokes.
* Few-shot identification of individuals in sports: the case of darts (2025):
  one recording identifies a thrower at about 80%, five at over 90%. Evidence
  that individual movement signatures are strong, which is the premise of
  within-player stroke classification.
* BFMD: a full-match badminton dense dataset for dense shot captioning (2026):
  19 broadcast matches including doubles, 1,687 rallies, 16,751 hit events
  with shot types, shuttle trajectories, poses and captions. Licence not
  stated in the abstract. A possible second source for elite bands and for
  doubles.
* Multisensory training enhances anticipation skills in badminton novices
  (Scientific Reports 2025); serve-specific gaze and kinematics studies
  reporting shot-related differences at shoulder, elbow, wrist and racket.
