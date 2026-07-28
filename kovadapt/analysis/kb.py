"""Cited aim-training knowledge base.

Distilled from primary sources by a cited research workflow; produced 2026-07-28. The source
base: motor-control literature (Fitts 1954; Guadagnoli & Lee 2004; Donovan & Radosevich 1999;
Walker et al. 2002; Adams 1961; Casiez & Vogel 2008; Donovan et al. 2022; Listman et al. 2021),
Aimer7's KovaaK's guide, the Voltaic documents and benchmark sheets, Viscose benchmark sheets,
and documented pro-player practice.

Standing rule this module serves: every insight kovadapt emits must carry its reasoning and its
sources ("cite everything"). Entries marked low-confidence, contested, or extrapolated must be
surfaced as such to the player — never presented as settled fact.

Contents:

- ``PRINCIPLES`` — coaching doctrine keyed by id (topic, text, confidence, sources).
- ``DIAGNOSTICS`` — telemetry-signal playbook keyed by id (signal, condition, interpretation,
  prescription, confidence, sources). ``condition`` is the human-readable rule; the engine
  implements rules in code and references these ids.
- ``BENCHMARKS`` — rank/threshold systems (Voltaic, Viscose, Aimer7 tiers, research anchors) in
  their researched structure. Numeric thresholds live in official sheets and drift over time —
  treat them as data to re-sync (see each entry's ``accuracy_caveat``), not ground truth.
- ``ROUTINES`` — training-routine templates and documented pro session structures.
- ``GAPS`` — honest holes in the source base; anything built past these is extrapolation.

Text is preserved verbatim from the research output (coach-grade prose with inline source
attributions); only line-wrapping was adjusted. Compound confidence labels such as
"high (pattern), medium (sens attribution)" are kept verbatim — the leading token is the
primary level.

Pure data module: zero imports (``kovadapt.analysis`` is a leaf package; core install stays
numpy-only).
"""

PRINCIPLES: dict[str, dict] = {
    "p-speed-accuracy-governor": {
        "topic": "Accuracy band governor (clicking)",
        "text": (
            "Hold click-timing accuracy inside an explicit 85-95% band and use it as a control "
            "loop: if a run drops below 85%, deliberately slow down; if it climbs above 95%, you "
            "are obligated to push speed. Accuracy is the constraint, speed is the variable you "
            "train. Voltaic states the same rule for static clicking (consistent 95%+ means "
            "increase pace, then rebuild to 95% again), and Aimlabs' guidance is that accuracy "
            "should come first, especially early, because starting fast builds sloppy habits that "
            "are much harder to unlearn. This is the doctrinal basis for kovadapt's accuracy "
            "deadband controller."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7, secs "
                "3.3-3.4)"
            ),
            (
                "https://docs.google.com/document/d/1JoNtoHK9GgJCjE-7yQxKXkpAkGJyOBBipiZqPNYwECs "
                "(Voltaic Aim Journey 4.1.1)"
            ),
            (
                "https://aimlabs.com/articles/aimlabs/the-speed-accuracy-tradeoff-and-what-it-means"
                "-for-your-aim-training/"
            ),
            (
                "https://docs.google.com/document/d/1TpFHOg6WbPS2iFie2z53AnyQTXQ_ZBg7lKNqjvWhXIE "
                "(Voltaic game-specific doc)"
            ),
        ),
    },
    "p-two-phase-flick": {
        "topic": "Two-phase flick model",
        "text": (
            "A flick decomposes into a primary ballistic movement toward the target — which lands "
            "hypermetric (overshoot) or hypometric (undershoot) — followed by one or more "
            "corrective submovements. This is the standard motor-control decomposition, validated "
            "on professional FPS players, and it is exactly what kovadapt's per-flick overshoot "
            "and corrective-submovement-count metrics implement. Expertise markers in the same "
            "study: faster reaction, higher movement speed, lower endpoint variability."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC9744923/ (Donovan et al. 2022, Front. "
                "Hum. Neurosci.)"
            ),
        ),
    },
    "p-swipiness": {
        "topic": "Shot timing along the flick ('swipiness')",
        "text": (
            "When you fire relative to the flick trajectory is a strategy dial, not a fixed "
            "virtue. 'Swipiness' near 0.5 means the shot lands mid-ballistic-movement (a swipe); "
            "at or above 1.0 the player flicks, lands, then fires. Skilled players deliberately "
            "lower swipiness (fire later) on precision tasks and raise it (fire earlier) on speed "
            "tasks. Consequence for diagnosis: overshoot with shots fired mid-movement and few "
            "corrections can be a legitimate speed strategy; overshoot followed by a chain of "
            "corrections is a control failure."
        ),
        "confidence": "high",
        "sources": (
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC9744923/ (Donovan et al. 2022)",
        ),
    },
    "p-microcorrect-not-reflick": {
        "topic": "Micro-correct, never re-flick",
        "text": (
            "Flicks are the least accurate movement type there is. After the initial flick your "
            "crosshair is already close to the target, so it is almost always better to "
            "micro-correct slowly and time the shot than to throw a second flick to show off. A "
            "second large ballistic correction after the first flick is a coachable flaw; the "
            "target pattern is one flick plus at most one small, slow correction."
        ),
        "confidence": "high",
        "sources": (
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7, p.5 fn.12)",
        ),
    },
    "p-switching-ideal": {
        "topic": "Switching ideal: zero micro-correction",
        "text": (
            "The target-switching ideal is landing the initial flick directly on the bot with no "
            "micro-correction at all, kept consistent on both small and large switches. In static "
            "clicking the standard is slightly looser: clean lines on the initial flick followed "
            "by micro-corrections that land directly on the bot — then diagnose whether accuracy "
            "or speed is the limiter and train that. Mean corrective submovements per acquisition "
            "is therefore a skill ladder: advanced switchers approach zero."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1vHiQRZMBJlmI69-SgHm3i0eS5ALfq2hEu-ZPyvC6ycE "
                "(Voltaic KovaaKs Benchmarks doc, VT psalmTS/skyTS and VT 1w6ts notes)"
            ),
        ),
    },
    "p-tracking-smoothness": {
        "topic": "Tracking doctrine: smoothness is skill, react don't predict",
        "text": (
            "For tracking, smoothness is synonymous with skill. Move in one smooth motion rather "
            "than many small motions, with minimal correcting. Do not predict the bot's movement — "
            "react to it, changing direction only when the target does, and never chase. "
            "Overprediction produces shaky aim and premature reactions. Keep visual focus on the "
            "bot, not the crosshair. In kovadapt terms: submovement count and velocity "
            "discontinuities during tracking are smoothness metrics, and overshoot clustered at "
            "target direction-changes is overprediction."
        ),
        "confidence": "high",
        "sources": (
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 fn.8)",
            (
                "https://docs.google.com/document/d/1JoNtoHK9GgJCjE-7yQxKXkpAkGJyOBBipiZqPNYwECs "
                "(Voltaic Aim Journey 4.1.2)"
            ),
        ),
    },
    "p-correction-anatomy": {
        "topic": "Tracking recovery: reaction part vs correcting part",
        "text": (
            "Tracking recovery splits into a reaction part (time from losing the target to the "
            "start of a correction) and a correcting part (time to re-acquire smooth tracking). "
            "The faster the correcting flick, the harder it is to resume proper tracking afterward "
            "— so below advanced level, correct smoothly even if it costs time. It is better to "
            "approach smoothly and undershoot than to overshoot and have to micro-adjust back. "
            "Correction latency and post-correction error are separate trainable quantities."
        ),
        "confidence": "high",
        "sources": (
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 sec 3.3)",
        ),
    },
    "p-fitts-throughput": {
        "topic": "Fitts's law and throughput as the fair skill score",
        "text": (
            "Movement time for aimed movements grows logarithmically with the index of difficulty "
            "ID = log2(D/W + 1): smaller and farther targets cost more time because the motor "
            "system has limited information capacity. Throughput (ID/MT) unifies speed and "
            "accuracy into one number and is the standard skill measure in motor research. "
            "kovadapt's flick duration-vs-amplitude regression is a Fitts fit: a falling "
            "ms-per-bit slope across sessions is genuine motor improvement, and per-run throughput "
            "is a fairer skill score than accuracy alone."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://www.semanticscholar.org/paper/58debe2530f83e392074dd7abab6339bdd1cc5b4 "
                "(Fitts 1954)"
            ),
            "https://www.yorku.ca/mack/chi2008-p1633.pdf (MacKenzie & Isokoski 2008)",
        ),
    },
    "p-speed-is-growth-axis": {
        "topic": "Long-term improvement shows up as speed, not accuracy",
        "text": (
            "In the largest longitudinal aim dataset (7,174 players, 682,564 runs), hit rate "
            "improved only modestly with practice while hits per second improved considerably, "
            "with 40-60% day-to-day retention. Expect a player's kills/s and flick speed to trend "
            "up over weeks while accuracy stays near their governed band. Rising accuracy with "
            "flat kills/s usually means the player is sandbagging speed, not improving."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://www.frontiersin.org/articles/10.3389/fnhum.2021.777779/full (Listman et "
                "al. 2021)"
            ),
        ),
    },
    "p-challenge-point": {
        "topic": "Train at the edge of ability (challenge point)",
        "text": (
            "Motor learning is maximized at an intermediate functional task difficulty relative to "
            "the learner's current skill: too-easy tasks carry no new information to learn, "
            "too-hard tasks exceed the capacity to use the information available. Difficulty must "
            "therefore be set per-player and per-session, not per-scenario — the direct scientific "
            "justification for kovadapt adapting scenario difficulty toward each player's edge. "
            "Voltaic's practical echo: never insist on a routine that is too hard for you; work up "
            "so you build the right habits, and timescale scenarios down when technique is failing."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://www.tandfonline.com/doi/full/10.1080/00222895.2025.2508283 (Guadagnoli & "
                "Lee 2004, Challenge Point Framework)"
            ),
            (
                "https://docs.google.com/document/d/1BPiDxbaqQVxInKwFfdr_AJdJDYVGKbUBGHJDKOeUy-Q "
                "(Voltaic fundamental routines)"
            ),
        ),
    },
    "p-spacing-and-sleep": {
        "topic": "Spaced practice and sleep consolidation",
        "text": (
            "Distributed practice reliably beats massed practice (meta-analysis of 63 studies: "
            "mean effect 0.46, largest for simple motor skills), and motor speed gains consolidate "
            "during sleep — roughly 20% overnight improvement where 12 waking hours produced none. "
            "The community arrived at the same place independently: consecutive training days are "
            "a better measure than total hours, useful volume saturates around 2-3 hours per day "
            "(split it if you train seriously), and improvement lands overnight. Voltaic's "
            "dedicated-aim-training recommendation is even shorter: about 30-40 minutes daily, at "
            "least 5 days a week, with the actual game as the rest of practice."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://gwern.net/doc/psychology/spaced-repetition/1999-donovan.pdf (Donovan & "
                "Radosevich 1999)"
            ),
            (
                "https://www.sciencedirect.com/science/article/pii/S0896627302007468 (Walker et "
                "al. 2002)"
            ),
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 sec 2)",
            (
                "https://blog.voltaic.gg/getting-started-with-voltaic/ (30-40 min guidance; medium "
                "confidence)"
            ),
        ),
    },
    "p-averages-not-highscores": {
        "topic": "Judge progress by averages and feel, not high scores",
        "text": (
            "High scores are strongly influenced by luck — your own variability, bot-pattern "
            "randomness, respawns — so never judge daily or weekly improvement by them. Average "
            "performance is a better indicator, and the best short-term signal is mouse-feel: "
            "motion becoming more solid, smoother, more reactive. Score plateaus while skill still "
            "improves are normal, and fluctuating between good and bad periods during improvement "
            "is expected. An insight engine should trend EWMAs and rolling averages, downweight "
            "single-run records, and message plateaus as normal."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 secs 2, "
                "3.6)"
            ),
        ),
    },
    "p-benchmarks-measure-routines-train": {
        "topic": "Benchmarks measure, routines train",
        "text": (
            "Benchmark scenarios are optimized for consistent scoring, not for practice — they "
            "measure skill; routines are what improve it. Benchmark about once a week, and spend "
            "training time on level-appropriate routines instead of grinding benchmark scenarios "
            "for rank. Rushing rank through a too-hard routine builds wrong habits."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1BPiDxbaqQVxInKwFfdr_AJdJDYVGKbUBGHJDKOeUy-Q "
                "(Voltaic fundamental routines)"
            ),
            (
                "https://docs.google.com/document/d/1vHiQRZMBJlmI69-SgHm3i0eS5ALfq2hEu-ZPyvC6ycE "
                "(Voltaic benchmarks doc)"
            ),
        ),
    },
    "p-weakness-isolation": {
        "topic": "Weakness isolation is the point of aim training",
        "text": (
            "The key advantage of an aim trainer over playing your game is isolation: you can "
            "simulate exactly the situations most beneficial to you and attack your weaknesses, "
            "whereas in real matches players unconsciously adopt play-styles that route around "
            "their mechanical weaknesses. Voltaic ships explicit weakness-specific routines, and "
            "its harmonic-mean energy system structurally forces balance — your rank is dragged "
            "down by your weakest subcategory, so training it is always the highest-yield move. "
            "This validates kovadapt targeting the measured weakest region, direction, and "
            "subskill."
        ),
        "confidence": "high",
        "sources": (
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 sec 1)",
            "https://blog.voltaic.gg/getting-started-with-voltaic/",
            "https://voltaic.medium.com/voltaic-kovaaks-benchmarks-season-4-35f3e3fb7512",
        ),
    },
    "p-contextual-interference": {
        "topic": "Vary and rotate scenarios (contextual interference)",
        "text": (
            "Random, varied practice ordering impairs today's scores but produces superior "
            "retention and transfer versus blocked grinding of one drill — the contextual "
            "interference effect, confirmed by a 2024 meta-analysis. Voltaic's routines apply this "
            "deliberately: alternate similar scenarios 'to avoid too much pattern recognition'. "
            "Rotating scenarios and varying difficulty (as kovadapt's adaptive variants do) is "
            "better learning than repeating one fixed scenario, even though the scoreboard looks "
            "worse in the moment."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://www.nature.com/articles/s41598-024-65753-3 (contextual interference "
                "meta-analysis; Shea & Morgan 1979)"
            ),
            (
                "https://docs.google.com/document/d/1BPiDxbaqQVxInKwFfdr_AJdJDYVGKbUBGHJDKOeUy-Q "
                "(Voltaic fundamental routines)"
            ),
        ),
    },
    "p-rest-position": {
        "topic": "Skill degrades away from the mouse rest position",
        "text": (
            "Everyone has a privileged rest position on the mousepad, and aiming skill is "
            "measurably better near it — the further from rest position, the worse the aim. This "
            "is why per-region deficits on the wall are expected rather than mysterious, and why "
            "they respond to dedicated large-angle training: 10 days to 2 weeks of 360-degree "
            "scenarios at 103-130 FOV, attempting never to lift the mouse. kovadapt's "
            "per-wall-region deficit targeting is this doctrine, automated."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1d1FY1qbwrgdj2K1wmhktbdcgG0ybeK3YJSeZ49yraW4 "
                "(Voltaic/Aimer7 issue-specific routines, Large Angles)"
            ),
        ),
    },
    "p-sensitivity-doctrine": {
        "topic": "Sensitivity: ranges, overshoot physics, and the contested stability rule",
        "text": (
            "Controlled research: high control-display gain increases overshooting (worst on small "
            "or distant targets) because motor-space distances shrink; very low gain instead costs "
            "time through clutching and limb-speed limits; performance is U-shaped with a broad "
            "usable middle. Community ranges sit inside that middle: tracking 20-25 cm/360, "
            "click-timing 30+ cm/360, versatile 21-27 cm/360 (Aimer7); Voltaic recommends 25-35 "
            "cm/360 with 20-50 acceptable, slower for flick-reliant games, faster for tracking "
            "games — and says not to obsess over it. On changing sensitivity: Voltaic calls the "
            "muscle-memory objection 'simplified and inaccurate' and endorses sens changes for "
            "practice as beneficial; Aimer7's rule is narrower — never change settings to inflate "
            "a score, but his own smoothness and speed protocols are deliberate temporary sens "
            "changes. Pro practice spans both poles (s1mple: one sens for years; TenZ: constant "
            "tiny tweaks), so sens stability is preference, not doctrine."
        ),
        "confidence": "high (physics and ranges); contested (stability rule)",
        "sources": (
            "https://gwern.net/doc/design/2008-casiez.pdf (Casiez & Vogel 2008)",
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 sec 3)",
            (
                "https://docs.google.com/document/d/1JoNtoHK9GgJCjE-7yQxKXkpAkGJyOBBipiZqPNYwECs "
                "(Aim Journey 2.3.1, 4.3.2)"
            ),
            "https://www.prosettings.com/s1mple-csgo-settings/",
            "https://pley.gg/valorant/tenz-valorant-settings/",
            (
                "https://www.vlr.gg/69967/open-letter-to-tenz-bro-dont-change-your-sens (folklore, "
                "low confidence)"
            ),
        ),
    },
    "p-limb-and-smoothness": {
        "topic": "Limb use vs sensitivity",
        "text": (
            "Lower sensitivities recruit more arm; higher sensitivities recruit more wrist and "
            "fingers. Arm-dominant low-sens players are generally smoother, because high sens "
            "scales every inconsistency and fingers offer the least control — but all good aimers "
            "use some wrist, so there is no reason to go extremely low either. Jittery traces at "
            "high sens and sluggish flicks at very low sens bracket the usable band from both "
            "sides."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1JoNtoHK9GgJCjE-7yQxKXkpAkGJyOBBipiZqPNYwECs "
                "(Aim Journey 4.3.2)"
            ),
            "https://x.com/vF_AIMER7/status/1230076886107250688",
        ),
    },
    "p-crosshair-placement": {
        "topic": "Crosshair placement beats raw aim for in-game transfer",
        "text": (
            "Raw mechanical training alone 'might win you a few fights' — crosshair placement and "
            "deathmatch win games. Keep the crosshair where an enemy can peek so that when they "
            "do, only a stop and a small micro-correction remain; a peek gives you a 300-400 ms "
            "window in which the target briefly stops, and micro-correction speed in that window "
            "is the trainable quantity. Being quick with placement is what removes the need for "
            "large-angle flicks at all. For kovadapt: large first-flick amplitudes in-game (not in "
            "the trainer) indicate placement problems, not mouse-control problems."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1TpFHOg6WbPS2iFie2z53AnyQTXQ_ZBg7lKNqjvWhXIE "
                "(Voltaic game-specific doc)"
            ),
            (
                "https://theguide.gg/improve/lessons/valorant/how-to-hold-angles-with-perfect-cross"
                "hair-placement-601 (secondary)"
            ),
        ),
    },
    "p-hit-confirm": {
        "topic": "Dynamic clicking: go to the target and hit-confirm",
        "text": (
            "On dynamic clicking, read each bot just long enough to confirm the hit — without "
            "being misdirected by a sudden direction change — and aim at targets directly rather "
            "than waiting for them to drift across a parked crosshair. Passive 'farming' of "
            "crosshair crossings inflates accuracy while stunting speed and reading. Timing cues "
            "matter too: click at the apex of a jump arc, and pace shots rather than spamming."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1vHiQRZMBJlmI69-SgHm3i0eS5ALfq2hEu-ZPyvC6ycE "
                "(VT Pasu/Bounceshot notes)"
            ),
            (
                "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 sec "
                "3.5, pasu)"
            ),
        ),
    },
    "p-warmup-decrement": {
        "topic": "Warmups work by reversing warm-up decrement",
        "text": (
            "The performance dip after a rest interval is warm-up decrement — a temporary, "
            "reversible set-loss distinct from forgetting — and a brief set-reinstating activity "
            "restores near-peak performance quickly. This is why short warmups work and long ones "
            "waste training budget: Voltaic caps warmup at about 15 minutes, and documented pro "
            "warmups run 15-40 minutes, deliberate, usually targeting one specific weakness per "
            "session (TenZ: one aspect of aim per deathmatch; yay and Ethos: weakness-first "
            "ordering, accuracy before speed)."
        ),
        "confidence": "high (mechanism); medium (pro details)",
        "sources": (
            (
                "https://www.semanticscholar.org/paper/953046c554c4d65978d4930b3fced0d3a6c30731 "
                "(Adams 1961)"
            ),
            "https://blog.voltaic.gg/getting-started-with-voltaic/",
            (
                "https://aimlabs.com/courses/7qomjBJ1VziOMBejmvHQ0a/lessons/48OoScMyHoZKBNlQfzfmQ9 "
                "(TenZ course)"
            ),
            "https://gamezo.gg/valorant-warmup-routines/",
        ),
    },
    "p-fatigue-rsi": {
        "topic": "Fatigue, overtraining, and injury",
        "text": (
            "Mechanics training saturates within a day — past roughly 2-3 useful hours, more is "
            "unproductive or counterproductive, and the gains you keep arrive overnight. "
            "Distinguish normal fine-motor muscle fatigue from tendon pain: at any suspicious "
            "pain, stop — RSI breaks the consecutive-days streak that actually drives improvement, "
            "and 1-2 week breaks can still return you improved. Even elites fail at this without "
            "external help: TenZ played through a hand injury until his team benched him to force "
            "rest; s1mple names the unbroken schedule as esports' biggest strain and protects one "
            "day off per week plus a 5-day break every two months. A worsening within-session "
            "trend is the signal to ease difficulty and end the session, not push through."
        ),
        "confidence": "high",
        "sources": (
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 sec 2)",
            (
                "https://dotesports.com/valorant/news/tenz-likely-to-play-vct-americas-match-throug"
                "h-hand-injury-with-sentinels-star-sub-still-out"
            ),
            "https://plarium.com/en/blog/interview-s1mple/",
        ),
    },
    "p-failure-taxonomy": {
        "topic": "Named failure modes for insights",
        "text": (
            "Voltaic's canonical list of things that can negatively affect aim is a ready-made "
            "label set for an insight engine: shakiness, slow reactivity, lack of speed, "
            "inaccuracy, overprediction, doubt, slow micro-corrections, poor timing/reading, poor "
            "target prioritization, poor crosshair placement. Pair every label with focused, "
            "purposeful practice on the identified weakness — deliberate practice beats mindless "
            "repetition."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1JoNtoHK9GgJCjE-7yQxKXkpAkGJyOBBipiZqPNYwECs "
                "(Aim Journey 4.4.3)"
            ),
        ),
    },
    "p-archetype-taxonomy": {
        "topic": "Archetype taxonomy alignment",
        "text": (
            "The community's canonical skill taxonomy matches kovadapt's archetypes exactly: "
            "Aimer7 splits all routines into tracking-dominated, click-timing-dominated, and "
            "complete, with target switching as the hybrid class 'because they contain both a "
            "flick and tracking part'; Voltaic formalizes Clicking / Tracking / Target Switching "
            "with subskills. Priorities per archetype: clicking = speed-accuracy balance with shot "
            "pacing; tracking = smoothness and reactivity; switching = flick speed, immediate "
            "stable track after landing, efficient pathing (acquire the target closest to your "
            "crosshair). Click-timing players must still train some tracking, because reading "
            "enemy movement transfers and is best trained by tracking. Shots-per-kill heuristics "
            "align: 1-3 click kills = clicking, held fire on long-TTK bots = tracking, short-TTK "
            "track after a flick = switching."
        ),
        "confidence": "high",
        "sources": (
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 sec 3)",
            (
                "https://blog.voltaic.gg/announcing-the-voltaic-season-5-aiming-benchmarks-beta-for"
                "-kovaaks/"
            ),
            (
                "https://docs.google.com/document/d/1JoNtoHK9GgJCjE-7yQxKXkpAkGJyOBBipiZqPNYwECs "
                "(Aim Journey 4.1.3)"
            ),
        ),
    },
    "p-accuracy-at-pace": {
        "topic": "Accuracy must be earned at pace (no accu-whoring)",
        "text": (
            "Accuracy goals only count 'without accu-whoring' — that is, while shooting for the "
            "whole scenario duration at pace, not by cherry-picking safe shots. Whole-run accuracy "
            "at speed is the honest number; an engine should compute and present it that way, and "
            "treat suspiciously high accuracy with low output as a red flag, not an achievement."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 secs "
                "3.5-3.7)"
            ),
        ),
    },
    "p-focus-strategy": {
        "topic": "Target-focus vs crosshair-focus (theory)",
        "text": (
            "Aimer7's stated theory, which he labels as such: focusing your eyes on the target "
            "trains reading and improves reactivity (better for close range and evasive enemies); "
            "focusing on the crosshair makes you smoother, more precise, and more aware of "
            "crosshair position (better for long range and click-timing accuracy). Use as a "
            "coaching lever — suggest target-focus for reactivity deficits and crosshair-focus for "
            "precision deficits — but present it as a theory to self-verify, as its author does."
        ),
        "confidence": "medium",
        "sources": (
            (
                "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 sec "
                "3.3, 'Theory')"
            ),
        ),
    },
    "p-scenario-quality": {
        "topic": "Prefer moving, varied targets; static grids are speed tools only",
        "text": (
            "Pure static grid scenarios (Tile Frenzy and kin) are poor for effective aim — "
            "effective click timing is only useful when targets move, and one-wall-one-target only "
            "tests visual reaction time. They earn a place only as dedicated speed-training tools "
            "at deliberately reduced sensitivity. Scenario recommendations should prefer moving, "
            "varied-target scenarios for real transfer."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 intro, "
                "sec 3.1)"
            ),
            "https://docs.google.com/document/d/1d1FY1qbwrgdj2K1wmhktbdcgG0ybeK3YJSeZ49yraW4",
        ),
    },
    "p-pro-training-spectrum": {
        "topic": "Pro training is a spectrum, not a recipe",
        "text": (
            "Documented pro practice spans aim-trainer-first (Demon1: daily Aimlabs routine, "
            "precision over spray), trainer-as-focused-warmup (TenZ: short playlist, then "
            "deathmatch focused on one aspect), hybrid DM-heavy (s1mple: ~30 min mixed warmup, "
            "then ~8 hours of team play), and in-game-only volume (donk: 10+ competitive maps "
            "daily, calls deathmatch duels 'unrealistic'; ZywOo: no individual practice at all, "
            "FACEIT and surf). Encode the spectrum rather than averaging it. The near-universal "
            "core: warmups are short, deliberate, and target one weakness at a time. Caution: pro "
            "aim-trainer endorsements are often paid (ZywOo's only Aim Lab tweet was tagged #ad) "
            "and can contradict stated habits."
        ),
        "confidence": "medium",
        "sources": (
            "https://aimlabs.com/articles/valorant/demon1s-ultimate-aim-training-routine/",
            "https://aimlabs.com/courses/7qomjBJ1VziOMBejmvHQ0a/lessons/48OoScMyHoZKBNlQfzfmQ9",
            "https://plarium.com/en/blog/interview-s1mple/",
            "https://esports.gg/news/counter-strike-2/donk-reveals-his-cs2-practice-routine/",
            "https://pley.gg/cs2/zywoo-dont-individual-practice-only-surf-kz-playing-fun/",
            "https://x.com/zywoo/status/1292855370667433985",
        ),
    },
    "p-composure": {
        "topic": "Pressure tolerance is trainable",
        "text": (
            "TenZ built his signature training tasks specifically to simulate in-game panic 'so "
            "players can practice being chill, calm, and precise' — composure under pressure is "
            "treated as a trainable aim skill, not a personality trait. A trainer can legitimately "
            "induce stress (timers, punish-on-miss) as a training stimulus, provided baseline "
            "difficulty stays at the challenge point."
        ),
        "confidence": "high",
        "sources": (
            "https://www.oneesports.gg/valorant/valorant-warm-up-routine-tenz-aim-lab/",
        ),
    },
}

DIAGNOSTICS: dict[str, dict] = {
    "dx-acc-above-band": {
        "signal": "accuracy vs archetype band",
        "condition": (
            "accuracy EWMA > archetype band ceiling (e.g. > 0.95 for clicking) sustained across "
            "recent runs"
        ),
        "interpretation": (
            "The player is sandbagging speed: pace is too comfortable and no new information is "
            "being learned (challenge point). Long-term skill growth shows up as speed, not "
            "accuracy — parked accuracy above the ceiling means the speed axis is idle."
        ),
        "prescription": (
            "Push pace: kovadapt's deadband controller should shrink targets / raise movement "
            "until accuracy re-enters the band. Coach line: 'Consistent 95%+? You are obligated to "
            "go faster, then rebuild to 95% at the new speed.'"
        ),
        "confidence": "high",
        "sources": (
            (
                "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 85-95% "
                "rule)"
            ),
            (
                "https://docs.google.com/document/d/1JoNtoHK9GgJCjE-7yQxKXkpAkGJyOBBipiZqPNYwECs "
                "(Voltaic 95%-then-speed)"
            ),
            "https://www.frontiersin.org/articles/10.3389/fnhum.2021.777779/full (Listman)",
        ),
    },
    "dx-acc-below-band": {
        "signal": "accuracy vs archetype band",
        "condition": "accuracy EWMA < archetype band floor (e.g. < 0.85 for clicking)",
        "interpretation": (
            "Over-driving speed. Below the floor the player is spraying flicks they cannot land, "
            "and speed-first play builds sloppy habits that are hard to unlearn."
        ),
        "prescription": (
            "Deliberately slow down until accuracy re-enters the band (kovadapt grows targets / "
            "calms movement). Accuracy first, then speed from a solid base. Whole-run accuracy at "
            "pace is the number that counts — no accu-whoring."
        ),
        "confidence": "high",
        "sources": (
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7)",
            (
                "https://aimlabs.com/articles/aimlabs/the-speed-accuracy-tradeoff-and-what-it-means"
                "-for-your-aim-training/"
            ),
        ),
    },
    "dx-overshoot-control": {
        "signal": "overshoot_rate + corrections",
        "condition": (
            "overshoot_rate > ~0.3 AND corrections >= 2 per flick on average, worst on small/far "
            "targets (numeric cutoffs are editorial calibration)"
        ),
        "interpretation": (
            "Hypermetric ballistic phase followed by a repair chain — a genuine control deficit, "
            "not a speed strategy. Research: high control-display gain increases overshooting, "
            "most on small/distant targets; community: over-flicking past the target then "
            "scrambling back is the classic flaw."
        ),
        "prescription": (
            "Slow the initial flick and aim to land just short, finishing with one slow "
            "micro-correction. If the pattern persists across sessions and is worst on small/far "
            "targets, trial a modest sensitivity reduction (the folk rule 'over-aim -> lower sens' "
            "matches the CD-gain research). Consider the smoothness/precision protocol."
        ),
        "confidence": "high (pattern), medium (sens attribution)",
        "sources": (
            "https://gwern.net/doc/design/2008-casiez.pdf (Casiez & Vogel 2008)",
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 fn.12)",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC9744923/ (Donovan)",
            (
                "https://steamcommunity.com/app/824270/discussions/0/1628538005513377948/ (folk "
                "rule, low)"
            ),
        ),
    },
    "dx-overshoot-strategic": {
        "signal": "overshoot_rate + corrections + shot timing",
        "condition": (
            "overshoot_rate elevated AND corrections low (<= 1) AND shots frequently fired "
            "mid-flick (before movement end) AND accuracy inside archetype band"
        ),
        "interpretation": (
            "This is the 'swipe' pattern — firing during the ballistic phase — which skilled "
            "players deliberately use on speed tasks. Overshoot without a correction chain and "
            "with banded accuracy is a speed strategy, not a control failure."
        ),
        "prescription": (
            "Do not 'fix' it. Leave difficulty on its current trajectory; only coach later shot "
            "timing if the player moves to precision-heavy scenarios where experts lower swipiness "
            "and fire later."
        ),
        "confidence": "high",
        "sources": (
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC9744923/ (Donovan et al., swipiness)",
        ),
    },
    "dx-undershoot-slow": {
        "signal": "mean_flick_ms vs amplitude (Fitts residual)",
        "condition": (
            "mean_flick_ms high for flick amplitude (positive Fitts residual / high ms-per-bit "
            "slope) AND overshoot_rate low AND accuracy at or above band"
        ),
        "interpretation": (
            "Under-driving: cautious hypometric flicks that creep to the target. Accurate but slow "
            "— the speed axis is the limiter. If the trace also shows clutching or very large arm "
            "sweeps, sensitivity may be too low (low CD gain costs time through clutching and "
            "limb-speed limits)."
        ),
        "prescription": (
            "Run the speed protocol: sensitivity divided by 2-4, 130 OW FOV, flick scenarios (Tile "
            "Frenzy variants, patTargetSwitch), holding fire the whole run — big slow-sens flicks "
            "make your real-sens flicks feel short. Then rebuild accuracy at the new pace inside "
            "the 85-95% band."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1d1FY1qbwrgdj2K1wmhktbdcgG0ybeK3YJSeZ49yraW4 "
                "(speed protocol)"
            ),
            "https://steamcommunity.com/sharedfiles/filedetails/?id=1818885969",
            "https://gwern.net/doc/design/2008-casiez.pdf (clutching cost)",
        ),
    },
    "dx-reflick": {
        "signal": "corrections (submovement amplitude profile)",
        "condition": (
            "secondary corrective movement amplitude comparable to the primary flick (a second "
            "ballistic movement instead of a micro-correction)"
        ),
        "interpretation": (
            "Re-flicking: after the first flick the crosshair is already near the target, and a "
            "second large flick is the least accurate way to close the gap. 'Almost always better "
            "to micro-correct slowly and time your shots than to do some idiotic flick.'"
        ),
        "prescription": (
            "Coach the one-flick-one-touch pattern: after the initial flick, finish with a single "
            "slow micro-correction and time the shot. In kovadapt, hold target size steady rather "
            "than shrinking, until the correction profile collapses to one small submovement."
        ),
        "confidence": "high",
        "sources": (
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 p.5 fn.12)",
        ),
    },
    "dx-tracking-jitter": {
        "signal": "corrections during tracking (smoothness)",
        "condition": (
            "archetype == tracking AND corrective submovement rate high / many velocity "
            "discontinuities during pursuit segments"
        ),
        "interpretation": (
            "Shaky tracking: many small motions instead of one smooth pursuit. Smoothness is "
            "synonymous with skill for tracking; jitter here is the smoothness deficit itself, not "
            "noise."
        ),
        "prescription": (
            "Smoothness/precision protocol: raise sensitivity 10-20% (forces smoothness because it "
            "is harder to control), 80 OW FOV, small dot crosshair, thin/small slow tracking "
            "scenarios 20-30 min/day for ~2 weeks. Cue: one smooth motion, minimal correcting; "
            "focus on the bot, not the crosshair."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1d1FY1qbwrgdj2K1wmhktbdcgG0ybeK3YJSeZ49yraW4 "
                "(smoothness protocol)"
            ),
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 fn.8)",
            "https://docs.google.com/document/d/1JoNtoHK9GgJCjE-7yQxKXkpAkGJyOBBipiZqPNYwECs",
        ),
    },
    "dx-overprediction": {
        "signal": "overshoot_rate clustered at target direction changes",
        "condition": (
            "archetype == tracking AND overshoot events concentrated at moments the target "
            "reverses/changes direction"
        ),
        "interpretation": (
            "Overprediction/overreaction: the player commits to where the bot was going instead of "
            "reacting to what it does, producing shaky aim and blown reversals. VT Air doctrine: "
            "'focus on not overreacting to changes in direction to optimize time on target'."
        ),
        "prescription": (
            "React-don't-predict cues; change direction only when the target does, never chase. "
            "For intermediate+ players, the reactivity protocol: 25-27 cm/360, 103 OW FOV, "
            "smallest dot (or crosshair off), thin fast-strafe scenarios, focus on the bot and "
            "reading. Below intermediate, prioritize smoothness fundamentals first and timescale "
            "scenarios down."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1JoNtoHK9GgJCjE-7yQxKXkpAkGJyOBBipiZqPNYwECs "
                "(Aim Journey 4.1.2)"
            ),
            (
                "https://docs.google.com/document/d/1d1FY1qbwrgdj2K1wmhktbdcgG0ybeK3YJSeZ49yraW4 "
                "(reactivity protocol)"
            ),
            (
                "https://docs.google.com/document/d/1vHiQRZMBJlmI69-SgHm3i0eS5ALfq2hEu-ZPyvC6ycE "
                "(VT Air)"
            ),
        ),
    },
    "dx-switch-corrections": {
        "signal": "corrections per target acquisition",
        "condition": "archetype == switching AND mean corrections per acquisition > 1",
        "interpretation": (
            "Below the switching ideal: the standard is landing the initial flick directly on the "
            "bot with zero micro-correction, consistent on small and large switches. A correction "
            "tax on every switch caps kills/s hard."
        ),
        "prescription": (
            "Slow the switch cadence and prioritize clean lines: first flick lands, then commit. "
            "Diagnose whether accuracy or speed is the limiter and train that one. Path "
            "efficiently — acquire the target closest to your crosshair. kovadapt should hold pace "
            "until corrections/acquisition approaches 1, then push speed."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1vHiQRZMBJlmI69-SgHm3i0eS5ALfq2hEu-ZPyvC6ycE "
                "(VT psalmTS/skyTS)"
            ),
            (
                "https://docs.google.com/document/d/1JoNtoHK9GgJCjE-7yQxKXkpAkGJyOBBipiZqPNYwECs "
                "(Aim Journey 4.1.3)"
            ),
        ),
    },
    "dx-bias": {
        "signal": "bias_score",
        "condition": (
            "abs(bias_score) sustained above threshold across runs (kovadapt convention: "
            "bias_score > 0 means the LEFT side is weaker)"
        ),
        "interpretation": (
            "Directional asymmetry — a real and common pattern (the community claims most "
            "right-handers struggle more tracking one direction; the prevalence claim is "
            "anecdotal, the prescription is primary)."
        ),
        "prescription": (
            "Bias practice toward the weak direction, exactly as kovadapt's dodge-direction skew "
            "does. Community form: 'if you have problems tracking to the right, circle the bot "
            "from the left more than from the right' — flip the drill so more engagements are in "
            "the weak direction until bias_score decays toward zero."
        ),
        "confidence": "medium",
        "sources": (
            (
                "https://docs.google.com/document/d/1d1FY1qbwrgdj2K1wmhktbdcgG0ybeK3YJSeZ49yraW4 "
                "(Controlled Tracking speed section)"
            ),
        ),
    },
    "dx-region-deficit": {
        "signal": "per-wall-region deficits",
        "condition": (
            "region posterior means > 0 concentrated in off-center regions (edges/corners of the "
            "wall, i.e. far from screen center and mouse rest position)"
        ),
        "interpretation": (
            "Skill decays with distance from the mousepad rest position — deficits at the wall's "
            "edges are the large-angle weakness, not randomness."
        ),
        "prescription": (
            "kovadapt's bandit already resamples spawns toward weak regions; reinforce with a "
            "large-angle block: 10 days-2 weeks of 360-style scenarios (Tile Frenzy 360 Strafing "
            "400%, Target Switching 360, LG pin practice 360) at 103-130 OW FOV, trying never to "
            "lift the mouse."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1d1FY1qbwrgdj2K1wmhktbdcgG0ybeK3YJSeZ49yraW4 "
                "(Large Angles)"
            ),
        ),
    },
    "dx-fatigue": {
        "signal": "session fatigue trend",
        "condition": (
            "fatigue trend positive: overshoot_rate AND mean_flick_ms worsening across the session "
            "(Theil-Sen slope up)"
        ),
        "interpretation": (
            "Within-session saturation. Mechanics training saturates from exhaustion; past that "
            "point more volume is unproductive or counterproductive, and the gains you keep "
            "consolidate overnight during sleep. Consecutive days beat marathon sessions."
        ),
        "prescription": (
            "Ease emitted difficulty (kovadapt already eases the plan without touching persisted "
            "state) and suggest a break or ending the session: 'gains land overnight — come back "
            "tomorrow.' Micro-breaks between runs help (pros use ~5-minute breaks between games). "
            "Hard stop at any tendon pain — RSI ends streaks."
        ),
        "confidence": "high",
        "sources": (
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 sec 2)",
            (
                "https://www.sciencedirect.com/science/article/pii/S0896627302007468 (Walker et "
                "al. 2002)"
            ),
            "https://gamezo.gg/valorant-warmup-routines/ (ScreaM micro-breaks, medium)",
        ),
    },
    "dx-input-health": {
        "signal": "jitter_ms / polling rate",
        "condition": (
            "timing jitter_ms high or effective polling rate well below the mouse's spec, or "
            "unstable during runs"
        ),
        "interpretation": (
            "Device/system noise, not player skill — and it degrades every telemetry inference "
            "downstream. Standard competitive setups run 1000 Hz+ polling (pro settings pages "
            "uniformly list 1000 Hz+). No primary community source defines numeric jitter cutoffs; "
            "kovadapt's thresholds are editorial."
        ),
        "prescription": (
            "Treat as a hardware/system fix, not a training item: check polling rate "
            "configuration, wireless interference/battery, background CPU load (kovadapt's checkup "
            "and game watchdog exist for exactly this). Suppress skill diagnoses derived from "
            "flick microstructure while input health is bad."
        ),
        "confidence": "medium (norm), low (cutoffs)",
        "sources": (
            "https://prosettings.net/players/donk/ (1000 Hz reference point)",
            "kovadapt-internal calibration (editorial)",
        ),
    },
    "dx-fitts-progress": {
        "signal": "mean_flick_ms vs amplitude trend across sessions",
        "condition": (
            "Fitts slope (ms per bit of difficulty) declining across sessions while accuracy and "
            "score EWMAs are flat"
        ),
        "interpretation": (
            "Genuine motor improvement hiding under a score plateau. Research: practice improves "
            "hits/second far more than hit rate; community: scores plateau while skill improves, "
            "and high scores are luck-noisy. The player is getting better even though the "
            "scoreboard disagrees."
        ),
        "prescription": (
            "Message the plateau as normal and show the throughput/Fitts trend as the progress "
            "metric. Keep trending EWMAs and averages; downweight PBs. Do not change the training "
            "plan on score stagnation alone."
        ),
        "confidence": "high",
        "sources": (
            "https://www.yorku.ca/mack/chi2008-p1633.pdf (throughput)",
            "https://www.frontiersin.org/articles/10.3389/fnhum.2021.777779/full (Listman)",
            (
                "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 secs 2, "
                "3.6)"
            ),
        ),
    },
    "dx-passive-farm": {
        "signal": "accuracy + kills/s + pre-shot movement",
        "condition": (
            "archetype == clicking (dynamic) AND accuracy high AND kills/s low AND little "
            "crosshair movement immediately before shots (targets crossing a parked crosshair)"
        ),
        "interpretation": (
            "Crosshair farming: waiting for targets to drift over the reticle instead of going to "
            "them. Inflates accuracy while stunting flick speed and reading — the slow-pasu drill "
            "exists specifically 'to learn to immediately go for the target, instead of waiting "
            "for it to cross your reticle'."
        ),
        "prescription": (
            "Coach: aim at targets directly; read the bot briefly to hit-confirm, then commit. In "
            "kovadapt, push pace (movement/spawn spread) so parking stops paying, and watch "
            "kills/s rather than accuracy for progress."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1BPiDxbaqQVxInKwFfdr_AJdJDYVGKbUBGHJDKOeUy-Q "
                "(slow pasu rationale)"
            ),
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 sec 3.5)",
            (
                "https://docs.google.com/document/d/1vHiQRZMBJlmI69-SgHm3i0eS5ALfq2hEu-ZPyvC6ycE "
                "(VT Pasu note)"
            ),
        ),
    },
    "dx-correction-speed": {
        "signal": "mean duration of post-flick corrective submovements",
        "condition": (
            "post-flick micro-correction durations long relative to the player's flick speed (slow "
            "settle after an accurate-enough flick)"
        ),
        "interpretation": (
            "'Slow micro-corrections' is a named failure mode: the flick is fine but the "
            "settle-and-confirm is slow, which in-game burns the 300-400 ms window a peeking enemy "
            "gives you."
        ),
        "prescription": (
            "Train micro-adjustment speed explicitly: small-angle flick then fastest-possible "
            "single correction, thinking about the next flick during the correction. "
            "Micro/precision scenarios (e.g. microcorrection-tagged drills) at game sens."
        ),
        "confidence": "high",
        "sources": (
            (
                "https://docs.google.com/document/d/1TpFHOg6WbPS2iFie2z53AnyQTXQ_ZBg7lKNqjvWhXIE "
                "(Valorant micro-correction window)"
            ),
            (
                "https://docs.google.com/document/d/1JoNtoHK9GgJCjE-7yQxKXkpAkGJyOBBipiZqPNYwECs "
                "(failure taxonomy)"
            ),
        ),
    },
}

BENCHMARKS: list[dict] = [
    {
        "system": "Voltaic KovaaK's Benchmarks",
        "version": (
            "Season 5 BETA (current season; announced 2024-12-24; sheet v1.0.7 still labeled BETA "
            "as of 2026-07-28; KovaaK's benchmark IDs Novice=459 Intermediate=458 Advanced=460; "
            "playlist sharecodes KovaaKsBouncingSilverBinding / KovaaKsBottingShinyDoor / "
            "KovaaKsBobbingSepiaBuff)"
        ),
        "categories": {
            "Clicking": ["Dynamic", "Static", "Linear (new S5 hybrid)"],
            "Tracking": ["Precise", "Reactive", "Control (new S5 hybrid)"],
            "Target Switching": ["Speed", "Evasive", "Stability (new S5 hybrid)"],
            "structure": (
                "3 tiers x 18 scenarios (3 categories x 3 subcategories x 2 scenarios). Ranks: "
                "Novice = Iron/Bronze/Silver/Gold (energy 100-400), Intermediate = "
                "Platinum/Diamond/Jade/Master (500-800), Advanced = "
                "Grandmaster/Nova/Astra/Celestial (900-1200)."
            ),
        },
        "scenarios": {
            "note": (
                "Static clicking shrinks target count with tier: VT 1w4ts (Novice) -> VT 1w3ts "
                "(Intermediate) -> VT 1w2ts (Advanced); all other families keep names with "
                "per-tier tuning."
            ),
            "archetype_mapping": (
                "Clicking subcats -> kovadapt 'clicking' (Static = small-target micro-flicks, "
                "Dynamic = large flicks to movers, Linear = strafing-target timing); Tracking "
                "subcats -> 'tracking' (Precise = slow smoothness, Reactive = erratic strafes, "
                "Control = high-speed wide); Switching subcats -> 'switching' (Speed = fast TS "
                "flicks, Evasive = dodging bots, Stability = track-then-switch hybrids). Mapping "
                "interpretation editorial (medium confidence)."
            ),
        },
        "thresholds": {
            "ranking_system": (
                "Energy per subcategory from your single best score, linear interpolation between "
                "adjacent rank targets; overall = HARMONIC MEAN of the 9 subcategory energies (0 "
                "until every subcategory has a score); rank = highest threshold met. Advanced "
                "scenario energy capped at 1200 until overall reaches Celestial, then uncapped for "
                "leaderboards (medium confidence). Note: the S4 doc's 'sum of subcategory energy' "
                "phrasing conflicts with this; resolved in favor of harmonic mean per the S5 "
                "Instructions tab verbatim text and app.voltaic.gg/leaderboards/about."
            ),
            "Novice (Iron/Bronze/Silver/Gold)": {
                "VT Pasu Novice S5": [555, 660, 745, 800],
                "VT Popcorn Novice S5": [390, 500, 600, 720],
                "VT 1w4ts Novice S5": [820, 915, 1010, 1110],
                "VT ww5t Novice S5": [990, 1090, 1190, 1290],
                "VT Frogtagon Novice S5": [620, 740, 850, 980],
                "VT Floating Heads Novice S5": [375, 460, 540, 640],
                "VT PGT Novice S5": [1900, 2325, 2775, 3050],
                "VT Snake Track Novice S5": [2400, 2750, 3125, 3425],
                "VT Aether Novice S5": [1525, 1900, 2250, 2650],
                "VT Ground Novice S5": [2100, 2500, 2825, 3100],
                "VT Raw Control Novice S5": [2125, 2550, 2975, 3450],
                "VT Controlsphere Novice S5": [1575, 1950, 2400, 2900],
                "VT DotTS Novice S5": [845, 940, 1030, 1090],
                "VT EddieTS Novice S5": [640, 730, 810, 890],
                "VT DriftTS Novice S5": [315, 355, 390, 430],
                "VT FlyTS Novice S5": [420, 460, 500, 535],
                "VT ControlTS Novice S5": [340, 380, 420, 450],
                "VT Penta Bounce Novice S5": [290, 340, 390, 445],
            },
            "Intermediate (Platinum/Diamond/Jade/Master)": {
                "VT Pasu Intermediate S5": [770, 850, 930, 980],
                "VT Popcorn Intermediate S5": [600, 690, 780, 860],
                "VT 1w3ts Intermediate S5": [1120, 1220, 1300, 1380],
                "VT ww5t Intermediate S5": [1310, 1400, 1490, 1560],
                "VT Frogtagon Intermediate S5": [940, 1040, 1140, 1230],
                "VT Floating Heads Intermediate S5": [610, 690, 770, 860],
                "VT PGT Intermediate S5": [2275, 2675, 3050, 3325],
                "VT Snake Track Intermediate S5": [2800, 3175, 3500, 3750],
                "VT Aether Intermediate S5": [2175, 2550, 2900, 3175],
                "VT Ground Intermediate S5": [2550, 2850, 3100, 3350],
                "VT Raw Control Intermediate S5": [2775, 3200, 3550, 3875],
                "VT Controlsphere Intermediate S5": [2750, 3175, 3525, 3825],
                "VT DotTS Intermediate S5": [1110, 1180, 1230, 1280],
                "VT EddieTS Intermediate S5": [880, 950, 1020, 1080],
                "VT DriftTS Intermediate S5": [390, 430, 460, 490],
                "VT FlyTS Intermediate S5": [520, 570, 610, 650],
                "VT ControlTS Intermediate S5": [420, 460, 485, 520],
                "VT Penta Bounce Intermediate S5": [450, 490, 540, 580],
            },
            "Advanced (Grandmaster/Nova/Astra/Celestial)": {
                "VT Pasu Advanced S5": [910, 1020, 1110, 1240],
                "VT Popcorn Advanced S5": [680, 800, 910, 1020],
                "VT 1w2ts Advanced S5": [1320, 1420, 1520, 1620],
                "VT ww5t Advanced S5": [1510, 1610, 1720, 1860],
                "VT Frogtagon Advanced S5": [1090, 1220, 1360, 1490],
                "VT Floating Heads Advanced S5": [740, 830, 920, 1050],
                "VT PGT Advanced S5": [2750, 3175, 3625, 4050],
                "VT Snake Track Advanced S5": [3050, 3425, 3725, 4050],
                "VT Aether Advanced S5": [2750, 3175, 3525, 3825],
                "VT Ground Advanced S5": [2875, 3200, 3500, 3725],
                "VT Raw Control Advanced S5": [3150, 3550, 3875, 4250],
                "VT Controlsphere Advanced S5": [3100, 3475, 3800, 4125],
                "VT DotTS Advanced S5": [1280, 1360, 1420, 1500],
                "VT EddieTS Advanced S5": [1020, 1120, 1200, 1280],
                "VT DriftTS Advanced S5": [430, 470, 510, 540],
                "VT FlyTS Advanced S5": [540, 600, 660, 720],
                "VT ControlTS Advanced S5": [450, 490, 520, 550],
                "VT Penta Bounce Advanced S5": [530, 580, 630, 670],
            },
            "progression_guidance": (
                "New players start Novice; reach Gold on Novice before Intermediate (aiming "
                "Platinum); reach Master on Intermediate before Advanced (aiming Grandmaster)."
            ),
        },
        "accuracy_caveat": (
            "High confidence: thresholds transcribed from the official sheet (gviz CSV, v1.0.7) "
            "and cross-verified against KovaaK's backend rank_maxes on 2026-07-28. Still "
            "officially a BETA — targets can be rebalanced. Scores are only comparable under "
            "Voltaic play rules (FOV 103+ OW equivalent, no pausing, latest scenario versions; VOD "
            "historically required for Grandmaster+ — rules corroborated indirectly, medium "
            "confidence). Voltaic energy is computed from SCORES, not accuracy; kovadapt's "
            "accuracy/telemetry metrics have no official mapping to energy. Machine-readable "
            "mirrors: kovaaks.com/webapp-backend benchmark endpoint (IDs above), Google Sheets "
            "gviz export, evxl.app static chunks."
        ),
    },
    {
        "system": "Voltaic KovaaK's Benchmarks",
        "version": (
            "Season 4 (historical, superseded by S5; sheet v1.2.9; KovaaK's IDs Novice=237 "
            "Intermediate=235 Advanced=253; sharecodes KovaaKsClearingPetulantScoutrifle / "
            "KovaaKsCrackingPrismaticPull / KovaaKsCrankingPulledGauntlet)"
        ),
        "categories": {
            "Clicking": ["Dynamic", "Static"],
            "Tracking": ["Precise", "Reactive"],
            "Switching": ["Speed", "Evasive"],
            "Strafe (Complete roles only, never counts toward base rank)": [
                "AngleStrafe",
                "ArcStrafe",
                "PatStrafe",
                "AirStrafe",
            ],
            "structure": (
                "6 core subcategories x 2 scenarios per tier; introduced the 'VT ' prefix and the "
                "energy system. Same 12-rank ladder as S5."
            ),
        },
        "scenarios": {
            "note": (
                "Static shrinks with tier: 1w6ts (Novice) -> 1w5ts (Intermediate) -> 1w3ts "
                "(Advanced); Multiclick 120 -> 120 -> 180."
            ),
        },
        "thresholds": {
            "Novice (Iron/Bronze/Silver/Gold)": {
                "VT Pasu Rasp Novice": [550, 650, 750, 850],
                "VT Bounceshot Novice": [500, 600, 700, 800],
                "VT 1w6ts Rasp Novice": [650, 750, 850, 950],
                "VT Multiclick 120 Novice": [1160, 1260, 1360, 1460],
                "VT Smoothbot Novice": [2300, 2500, 3100, 3500],
                "VT PreciseOrb Novice": [1300, 1600, 1900, 2200],
                "VT Plaza Novice": [2150, 2450, 2850, 3050],
                "VT Air Novice": [1900, 2200, 2500, 2800],
                "VT psalmTS Novice": [620, 690, 760, 830],
                "VT skyTS Novice": [780, 860, 950, 1040],
                "VT evaTS Novice": [450, 510, 560, 620],
                "VT bounceTS Novice": [490, 550, 610, 680],
            },
            "Intermediate (Platinum/Diamond/Jade/Master)": {
                "VT Pasu Rasp Intermediate": [750, 850, 950, 1050],
                "VT Bounceshot Intermediate": [600, 700, 800, 900],
                "VT 1w5ts Rasp Intermediate": [1000, 1100, 1200, 1300],
                "VT Multiclick 120 Intermediate": [1360, 1460, 1560, 1660],
                "VT Smoothbot Intermediate": [3050, 3450, 3850, 4250],
                "VT PreciseOrb Intermediate": [1650, 2050, 2450, 2850],
                "VT Plaza Intermediate": [2680, 2980, 3280, 3530],
                "VT Air Intermediate": [2450, 2700, 2950, 3200],
                "VT psalmTS Intermediate": [810, 880, 950, 1020],
                "VT skyTS Intermediate": [1030, 1130, 1220, 1300],
                "VT evaTS Intermediate": [550, 600, 650, 700],
                "VT bounceTS Intermediate": [630, 670, 710, 760],
                "VT AngleStrafe Intermediate (Complete only)": [740, 830, 920, 1000],
                "VT ArcStrafe Intermediate (Complete only)": [660, 750, 850, 940],
                "VT PatStrafe Intermediate (Complete only)": [2260, 2620, 2800, 3050],
                "VT AirStrafe Intermediate (Complete only)": [2800, 3000, 3200, 3400],
            },
            "Advanced (Grandmaster/Nova/Astra/Celestial)": {
                "VT Pasu Rasp Advanced": [940, 1040, 1120, 1270],
                "VT Bounceshot Advanced": [800, 900, 1000, 1150],
                "VT 1w3ts Rasp Advanced": [1280, 1380, 1460, 1580],
                "VT Multiclick 180 Advanced": [1630, 1770, 1890, 2000],
                "VT Smoothbot Advanced": [3300, 3600, 3950, 4300],
                "VT PreciseOrb Advanced": [2500, 2850, 3250, 3650],
                "VT Plaza Advanced": [3275, 3475, 3600, 3800],
                "VT Air Advanced": [3000, 3250, 3500, 3750],
                "VT psalmTS Advanced": [1080, 1160, 1200, 1330],
                "VT skyTS Advanced": [1300, 1430, 1500, 1600],
                "VT evaTS Advanced": [680, 740, 780, 830],
                "VT bounceTS Advanced": [820, 920, 970, 1050],
                "VT AngleStrafe Advanced (Complete only)": [880, 1020, 1150, 1230],
                "VT ArcStrafe Advanced (Complete only)": [940, 1080, 1150, 1230],
                "VT PatStrafe Advanced (Complete only)": [3050, 3240, 3400, 3500],
                "VT AirStrafe Advanced (Complete only)": [3400, 3600, 3700, 3825],
            },
        },
        "accuracy_caveat": (
            "High confidence transcription from official S4 sheet + KovaaK's backend. Historical — "
            "use only for interpreting old scores. MBS strafe scoring is capped (200 "
            "Angle/ArcStrafe, 1000 AirStrafe; PatStrafe conditional distance scoring). Settings "
            "rules: min 103 OW FOV, 25-35 cm/360 recommended, video evidence for Grandmaster+."
        ),
    },
    {
        "system": "Voltaic KovaaK's Benchmarks",
        "version": "Season 3 (June 2021, historical) and lineage",
        "categories": {
            "structure": (
                "Two tiers, threshold ('basic') ranking, no energy: Intermediate (KovaaK's id 265; "
                "Bronze/Silver/Gold/Platinum/Diamond) and Advanced (id 266; "
                "Jade/Master/Grandmaster/Nova/Astra/Celestial). Scenario families: Pasu "
                "(sqrt-accuracy scoring), voxTS, patTS, ww3t, B180(T), 1w6ts etc., pre-'VT ' "
                "prefix."
            ),
        },
        "scenarios": {
            "lineage": (
                "Original Voltaic progression benchmarks date to 2019 (Steam guide era); S3 "
                "(2021-06-05) added Jade, Astra (replacing Ascended), Celestial, Bronze targets, "
                "VOD requirements for GM+; S4 (mid-2022) introduced VT prefix + energy/harmonic "
                "mean; S5 BETA 2024-12-24. Community 'Elite (Unofficial)' extensions exist above "
                "Celestial (S4 id 1977: Ascended/Eclipse; S5 id 475: Stellaris/Lunara/Solara) — "
                "NOT official."
            ),
        },
        "thresholds": {
            "note": "Per-scenario S3 thresholds not transcribed in this research pass.",
        },
        "accuracy_caveat": (
            "Medium confidence (lineage reconstructed from multiple sources including evxl.app "
            "rankCalculation fields). Historical context only."
        ),
    },
    {
        "system": "Voltaic KovaaK's Benchmarks",
        "version": (
            "S5.5 Advanced-only TEST revision (KovaaK's id 2070) — unannounced, no official sheet; "
            "treat as in-testing, NOT the current official season"
        ),
        "categories": {
            "structure": (
                "Advanced tier only, ranks Grandmaster/Nova/Astra/Celestial; revises scenarios "
                "(Viscose/Ava variants) and expands Reactive to 3 and Speed-switching to 4 "
                "scenarios; VT AvasiveTS replaces VT DriftTS."
            ),
        },
        "scenarios": {
            "changed": [
                "VT Pasu Viscose Advanced S5",
                "VT Popcorn Ava Advanced S5",
                "VT Air Spectral Advanced S5 Test 2",
                "VT Controlsphere Ava Advanced S5",
                "VT EddieTS Viscose Advanced S5",
                "VT YoxTS Advanced S5",
                "VT voxTS Advanced S5",
                "VT AvasiveTS Advanced S5",
            ],
        },
        "thresholds": {
            "Advanced (GM/Nova/Astra/Celestial)": {
                "VT Pasu Viscose Advanced S5": [910, 1020, 1110, 1240],
                "VT Popcorn Ava Advanced S5": [670, 800, 900, 1020],
                "VT 1w2ts Advanced S5": [1320, 1420, 1520, 1620],
                "VT ww5t Advanced S5": [1510, 1610, 1720, 1860],
                "VT Frogtagon Advanced S5": [1090, 1220, 1360, 1490],
                "VT Floating Heads Advanced S5": [740, 830, 920, 1050],
                "VT PGT Advanced S5": [2750, 3175, 3625, 4050],
                "VT Snake Track Advanced S5": [3050, 3425, 3725, 4050],
                "VT Aether Advanced S5": [2750, 3175, 3525, 3825],
                "VT Air Spectral Advanced S5 Test 2": [3800, 4400, 4950, 5500],
                "VT Ground Advanced S5": [2875, 3200, 3500, 3725],
                "VT Raw Control Advanced S5": [3150, 3550, 3875, 4250],
                "VT Controlsphere Ava Advanced S5": [3150, 3500, 3925, 4200],
                "VT DotTS Advanced S5": [1280, 1360, 1420, 1500],
                "VT EddieTS Viscose Advanced S5": [1020, 1120, 1200, 1270],
                "VT YoxTS Advanced S5": [90, 105, 110, 125],
                "VT voxTS Advanced S5": [105, 117, 130, 140],
                "VT AvasiveTS Advanced S5": [500, 550, 610, 680],
                "VT FlyTS Advanced S5": [540, 600, 660, 720],
                "VT ControlTS Advanced S5": [450, 490, 520, 550],
                "VT Penta Bounce Advanced S5": [530, 580, 630, 670],
            },
        },
        "accuracy_caveat": (
            "Medium confidence: pulled from live KovaaK's API + evxl.app label 'Voltaic S5.5'; no "
            "official Voltaic announcement or spreadsheet found. Could change or never ship."
        ),
    },
    {
        "system": "Viscose Benchmarks ('The Viscose Benches')",
        "version": (
            "2025 beta (a.k.a. S1; released 2025-07-06/07; by Viscose/'lawa' with pinguefy, Raw "
            "Input Discord; self-described 'a practice playlist disguised as a benchmark'; "
            "KovaaK's IDs Easier=686 Medium=687 Hard=688)"
        ),
        "categories": {
            "Control Tracking": ["Arm x3", "Wrist x3", "Fingertip x3", "Blending x3"],
            "Reactive Tracking": ["Control x2", "Speed x2", "Reading x2"],
            "Flick Tech": ["Speed x3", "Stability x2", "Micro x2", "Evasive/Post-flick x2"],
            "Dynamic Clicking": ["Reading x3", "Precision x3", "Linear/Stability x3"],
            "structure": (
                "3 tabs x 36 scenarios. Unique feature: Arm/Wrist/Fingertip subcats categorize by "
                "the MUSCLE GROUP stressed — directly usable to phrase kovadapt deficit "
                "prescriptions. Tab difficulty describes scenario difficulty, not player skill; "
                "Medium ranks 'stretch well into what would be exclusive to the advanced "
                "benchmarks on voltaic'."
            ),
        },
        "scenarios": {
            "rank_ladders": {
                "Easier (animals, 8)": [
                    "Lemming",
                    "Hare",
                    "Ermine",
                    "Penguin",
                    "Fox",
                    "Mammoth",
                    "Orca",
                    "Seal",
                ],
                "Medium (colors, 8)": [
                    "Cinnabar",
                    "Vermillion",
                    "Saffron",
                    "Celadon",
                    "Cerulean",
                    "Lavender",
                    "Indigo",
                    "Fuchsia",
                ],
                "Hard (fabrics, 6)": ["Wool", "Linen", "Velvet", "Chiffon", "Satin", "Silk"],
            },
            "rank_computation": (
                "Ranked once every subcategory has a score; tier rank = LOWEST subcategory rank; "
                "subcategory rank = highest threshold met by its best scenario. ' Complete' suffix "
                "when every scenario individually reaches the overall rank. 'Threads' = cosmetic "
                "points, round(100*(score-lowest)/(highest-lowest)) floored at 0, summed over 36 "
                "scenarios."
            ),
            "validity_rules": (
                "Fingertip scores invalid slower than 50cm/360; 103 hfov minimum everywhere. "
                "Progression: try Medium at comfortable Mammoth; try Hard around Lavender/Indigo."
            ),
        },
        "thresholds": {
            "Easier (8 ranks Lemming->Seal)": {
                "WhisphereRawControl Larger + Slowed": [
                    5500,
                    6700,
                    7800,
                    8700,
                    9600,
                    10500,
                    11400,
                    12500,
                ],
                "Whisphere 80%": [6300, 7700, 9000, 10000, 11000, 12000, 13000, 14500],
                "SmoothBot Invincible Goated 75%": [1800, 2250, 2650, 2900, 3150, 3400, 3650, 4000],
                "Leaptrack Goated 60% Larger": [850, 1200, 1500, 1700, 1900, 2100, 2250, 2450],
                "Controlsphere rAim Easy 90%": [6100, 6950, 7700, 8400, 9100, 9800, 10500, 11500],
                "VT Controlsphere Intermediate S5 80%": [
                    1850,
                    2300,
                    2700,
                    3000,
                    3300,
                    3600,
                    3850,
                    4100,
                ],
                "Air Angelic 4 Voltaic Easy 80% (Good Version)": [
                    1050,
                    1600,
                    2000,
                    2400,
                    2700,
                    3000,
                    3300,
                    3600,
                ],
                "cloverRawControl Easy 80% Speed": [3900, 4550, 5200, 5700, 6200, 6700, 7200, 7700],
                "Controlsphere Far, Far Larger 90%": [
                    7600,
                    8150,
                    8700,
                    9200,
                    9800,
                    10200,
                    10900,
                    11500,
                ],
                "PGTI Voltaic Easy 80%": [350, 550, 850, 1100, 1350, 1600, 1900, 2250],
                "Air CELESTIAL No UFO Easy Slowed": [820, 835, 850, 861, 870, 878, 884, 890],
                "Whisphere Small & Slow 55%": [6000, 7500, 9000, 10000, 10750, 11500, 12250, 13500],
                "Air Voltaic Invincible 7 Easy 80%": [
                    750,
                    1200,
                    1600,
                    1900,
                    2200,
                    2500,
                    2800,
                    3200,
                ],
                "Controlsphere OW Long Strafes 90%": [
                    5400,
                    6100,
                    6700,
                    7200,
                    7600,
                    8000,
                    8300,
                    8700,
                ],
                "Flicker Plaza rAim Easy Less Blinks": [858, 871, 883, 890, 895, 900, 904, 909],
                "Polarized Hell Easy 40% Slower": [750, 1100, 1400, 1600, 1800, 2000, 2150, 2500],
                "Air Pure Intermediate Slower No UFO": [860, 874, 886, 893, 901, 907, 911, 916],
                "Air Voltaic Easy Invincible 4 80%": [
                    1100,
                    1600,
                    2100,
                    2450,
                    2800,
                    3150,
                    3400,
                    3800,
                ],
                "Pokeball Frenzy Auto TE Wide": [650, 950, 1250, 1500, 1750, 2000, 2300, 2700],
                "1w3ts Reload Larger": [36, 43, 50, 58, 70, 82, 92, 102],
                "voxTargetSwitch 2 Large": [67, 78, 87, 95, 103, 110, 117, 123],
                "beanTS Larger": [65, 78, 90, 100, 110, 120, 130, 142],
                "FloatTS Angelic Easy Larger": [65, 74, 81, 88, 95, 101, 106, 111],
                "waldoTS Novice": [65, 78, 90, 100, 110, 120, 130, 140],
                "devTS Goated NR Static 5Bot": [350, 400, 450, 500, 550, 600, 640, 680],
                "domiSwitch Easy Slower": [3200, 3700, 4200, 4600, 5000, 5400, 5800, 6300],
                "tamTargetSwitch Smooth Easy": [7, 11, 15, 18, 21, 24, 26, 28],
                "1wall5targets_pasu slow": [76, 88, 100, 110, 120, 130, 140, 150],
                "B180 Voltaic Easy 92%": [26, 38, 50, 58, 65, 72, 78, 87],
                "Controlsphere Click Easy": [15, 21, 27, 33, 39, 45, 50, 55],
                "Popcorn MV Novice": [50, 100, 150, 190, 230, 270, 300, 330],
                "Pasu Angelic 20% Larger 80% Speed": [51, 58, 65, 72, 78, 84, 90, 97],
                "1w2ts Pasu Perfected Easy": [58, 69, 80, 87, 93, 99, 105, 110],
                "1w3ts Pasu Perfected Micro Goated Larger 80%": [
                    600,
                    700,
                    800,
                    900,
                    1000,
                    1100,
                    1200,
                    1300,
                ],
                "Floating Heads Timing 400% Larger": [400, 700, 1000, 1350, 1700, 2050, 2400, 2750],
                "voxTargetSwitch Click": [49, 59, 67, 74, 81, 88, 94, 100],
            },
            "Medium (8 ranks Cinnabar->Fuchsia)": {
                "WhisphereRawControl": [5300, 6700, 7600, 8800, 10000, 10900, 11800, 13000],
                "Whisphere": [5750, 8250, 10750, 13250, 15250, 17250, 19250, 20500],
                "Smoothbot Invincible Goated": [1700, 2200, 2750, 3250, 3800, 4300, 4650, 5000],
                "Leaptrack Goated 75% Slightly Larger": [
                    1400,
                    1750,
                    2000,
                    2250,
                    2500,
                    2750,
                    3000,
                    3275,
                ],
                "Controlsphere rAim Easy": [7400, 8500, 9600, 10700, 11900, 13100, 14300, 15100],
                "VT Controlsphere Novice S5 Hard": [2100, 2500, 2950, 3250, 3550, 3800, 4100, 4400],
                "Air Angelic 4 Voltaic Easy": [1900, 2300, 2650, 3000, 3400, 3750, 4100, 4450],
                "cloverRawControl Easy": [5000, 6100, 7000, 8100, 9200, 10100, 10900, 11700],
                "Controlsphere Far Larger": [7800, 8600, 9300, 10000, 10800, 11700, 12700, 13600],
                "PGTI Voltaic Easy": [800, 1100, 1400, 1700, 2000, 2300, 2750, 3200],
                "Air CELESTIAL No UFO Easy": [825, 840, 855, 865, 881, 890, 902, 908],
                "Whisphere Small & Slow 75%": [5500, 7500, 9500, 11500, 13500, 15500, 17500, 19000],
                "Ground Plaza Sparky v3": [862, 872, 882, 888, 894, 900, 905, 909],
                "Controlsphere OW": [4800, 5700, 6400, 7300, 8100, 8900, 9800, 10500],
                "Flicker Plaza rAim Easy": [860, 870, 881, 891, 900, 908, 913, 917],
                "Polarized Hell Easy 20% Slower": [1600, 1850, 2100, 2350, 2600, 2850, 3150, 3350],
                "Air Pure Intermediate": [847, 862, 876, 886, 895, 902, 906, 910],
                "Air Voltaic Easy Invincible 4": [1800, 2150, 2500, 2950, 3350, 3750, 4150, 4450],
                "Pokeball Frenzy Auto TE Wide": [1950, 2250, 2550, 2850, 3150, 3400, 3600, 3800],
                "1w3ts Reload": [66, 76, 86, 96, 106, 116, 126, 135],
                "voxTargetSwitch 2": [78, 88, 98, 107, 116, 123, 130, 136],
                "beanTS": [88, 103, 115, 127, 136, 143, 149, 156],
                "FloatTS Angelic Easy": [70, 79, 86, 93, 100, 107, 115, 123],
                "waldoTS Intermediate": [108, 117, 126, 135, 144, 153, 162, 170],
                "devTS Goated NR Static 5Bot": [600, 650, 705, 760, 810, 840, 870, 900],
                "domiSwitch Easy": [4200, 4700, 5200, 5700, 6150, 6600, 7100, 7600],
                "tamTargetSwitch Smooth Easy": [22, 26, 29, 32, 34, 36, 38, 41],
                "1wall5targets_pasu Reload": [70, 85, 100, 115, 130, 142, 155, 165],
                "VT Bounceshot Intermediate": [550, 640, 720, 780, 850, 900, 980, 1060],
                "Controlsphere Click": [27, 33, 39, 45, 50, 56, 61, 67],
                "Popcorn MV Intermediate": [150, 190, 240, 280, 330, 380, 430, 480],
                "Pasu Angelic 20% Larger": [72, 79, 85, 90, 96, 103, 110, 115],
                "1w2ts Pasu Perfected": [60, 70, 80, 88, 96, 101, 107, 112],
                "1w3ts Pasu Perfected Micro Goated Larger": [
                    900,
                    1000,
                    1100,
                    1200,
                    1300,
                    1400,
                    1500,
                    1600,
                ],
                "Floating Heads Timing 400% Larger": [
                    1950,
                    2300,
                    2650,
                    3000,
                    3350,
                    3650,
                    3900,
                    4200,
                ],
                "voxTargetClick 20% Small": [62, 72, 80, 87, 95, 102, 108, 116],
            },
            "Hard (6 ranks Wool->Silk)": {
                "WhisphereRawControl 30% Small": [7100, 7700, 8300, 8900, 9900, 11000],
                "Whisphere Small & Slow": [14000, 15250, 16500, 18000, 19500, 20500],
                "SmoothBot Invincible Goated Smaller": [2700, 3200, 3450, 3850, 4200, 4450],
                "Leaptrack Goated 80%": [2500, 2625, 2750, 2850, 3000, 3200],
                "Controlsphere rAim": [9100, 9850, 10700, 11300, 11900, 12350],
                "VT Controlsphere Intermediate Hard": [2850, 3200, 3500, 3800, 4000, 4200],
                "Air Angelic 4 Voltaic": [3300, 3650, 3850, 4100, 4225, 4350],
                "cloverRawControl": [7000, 7650, 8200, 8650, 9200, 10100],
                "Controlsphere Far": [8600, 9200, 9800, 10600, 11200, 12000],
                "PGTI Voltaic": [1350, 1650, 1900, 2150, 2400, 2700],
                "Air CELESTIAL": [867, 875, 885, 890, 894, 897],
                "Whisphere Extra Small & Slow": [8500, 10300, 11000, 12500, 13500, 14500],
                "Ground Plaza Sparky v3 Thin": [881, 886, 891, 895, 898, 901],
                "Controlsphere OW 150%": [6900, 7600, 8100, 8525, 8950, 9500],
                "Flicker Plaza": [890, 896, 901, 905, 910, 914],
                "Polarized Hell 20% Slower": [2550, 2700, 2850, 3000, 3150, 3300],
                "Air Pure": [884, 890, 895, 900, 905, 909],
                "Air Voltaic Invincible 4": [2800, 3150, 3450, 3700, 3900, 4100],
                "Pokeball Frenzy Auto TE Wide": [3550, 3725, 3850, 4000, 4100, 4200],
                "1w2ts Reload": [106, 114, 121, 127, 133, 138],
                "voxTargetSwitch 2 20% Smaller": [103, 111, 116, 121, 127, 133],
                "beanTS 30% Smaller": [119, 127, 134, 139, 143, 147],
                "FloatTS Angelic": [94, 100, 105, 110, 114, 118],
                "waldoTS": [145, 153, 160, 166, 173, 178],
                "devTS Goated NR Static Small 5Bot": [750, 775, 800, 825, 850, 870],
                "domiSwitch": [5550, 5950, 6250, 6550, 6850, 7200],
                "tamTargetSwitch Smooth": [32, 35, 37, 39, 42, 45],
                "Pasu Reload Goated": [110, 120, 130, 140, 150, 160],
                "VT Bounceshot Advanced": [730, 790, 850, 910, 950, 1000],
                "Controlsphere Click Smaller": [39, 45, 51, 56, 60, 64],
                "Popcorn MV Advanced": [290, 330, 370, 420, 460, 500],
                "Pasu Angelic": [87, 94, 102, 110, 118, 125],
                "1w2ts Pasu Perfected 30% Smaller": [75, 82, 87, 93, 98, 103],
                "1w3ts Pasu Perfected Micro Goated": [1100, 1200, 1300, 1400, 1500, 1560],
                "Floating Heads Timing 400% Fixed": [3200, 3484, 3648, 3848, 4048, 4248],
                "VoxTargetSwitch Click Small": [90, 96, 101, 106, 111, 115],
            },
        },
        "accuracy_caveat": (
            "High confidence transcription (sheet CSV/xlsx export + KovaaK's API id 686 verified). "
            "BUT the 2025 beta explicitly labels its targets 'prospective' (balanced via feedback "
            "form) — expect drift. Superseded by Viscose S2 in 2026-04. Hard-tier Silk targets sit "
            "at/near the creator's own PBs."
        ),
    },
    {
        "system": "Viscose Benchmarks",
        "version": (
            "S2 (2026; playable 2026-04-27; home = evxl.app; KovaaK's IDs Easier=2335 Medium=2336 "
            "Hard=2337 Expert=2338; playlists 'Viscose Benchmark S2 - Easier/Medium/Hard/Expert'; "
            "sheet 1WeuEk444WOkTpvOGMYiertxwlI9gRQSapYiwxjFbT08)"
        ),
        "categories": {
            "Control Tracking": ["Arm x3", "Wrist x3", "Fingertip x3", "Blending x3"],
            "Reactive Tracking": ["Control x2", "Speed x2", "Reading x2"],
            "Target Switching / Flick Tech": [
                "Speed x3",
                "Stability x3",
                "Micro x3",
                "Post-Flick x3 (new)",
            ],
            "Dynamic Clicking": ["Reading x3", "Precision x3", "Stability x3"],
            "structure": (
                "4 tiers x 39 scenarios, 14 subcategories. Easier rank thresholds explicitly "
                "calibrated to leaderboard percentiles: Lemming 99%, Hare 96%, Ermine 91%, Puffin "
                "86%, Penguin 80%, Fox 72%, Mammoth 60%, Orca 48%, Seal 37-38%."
            ),
        },
        "scenarios": {
            "rank_ladders": {
                "Easier (9)": [
                    "Lemming",
                    "Hare",
                    "Ermine",
                    "Puffin",
                    "Penguin",
                    "Fox",
                    "Mammoth",
                    "Orca",
                    "Seal",
                ],
                "Medium (9)": [
                    "Cinnabar",
                    "Vermillion",
                    "Saffron",
                    "Celadon",
                    "Viridian",
                    "Cerulean",
                    "Lavender",
                    "Indigo",
                    "Fuchsia",
                ],
                "Hard (8)": [
                    "Wool",
                    "Rayon",
                    "Linen",
                    "Velvet",
                    "Chiffon",
                    "Tricot",
                    "Satin",
                    "Silk",
                ],
                "Expert (6)": ["Interloper", "Attuned", "Heroic", "Mythic", "Ascension", "Eclipse"],
            },
        },
        "thresholds": {
            "Easier (9 ranks Lemming->Seal)": {
                "Smoothsphere Viscose Easier": [
                    5500,
                    6500,
                    8050,
                    9200,
                    10200,
                    11400,
                    12300,
                    13050,
                    13600,
                ],
                "Whisphere Viscose Easier": [
                    6800,
                    9400,
                    10300,
                    11100,
                    12800,
                    14500,
                    16200,
                    17500,
                    18500,
                ],
                "SmoothBot Perfected Easier": [
                    2000,
                    2500,
                    3000,
                    3350,
                    3650,
                    4100,
                    4500,
                    4775,
                    5025,
                ],
                "Leapstrafes Control wobin Easier": [
                    1200,
                    1600,
                    2025,
                    2250,
                    2450,
                    2750,
                    3000,
                    3250,
                    3450,
                ],
                "Controlsphere SuperbAim Viscose Easier": [
                    4200,
                    5100,
                    6200,
                    7050,
                    7975,
                    9050,
                    10000,
                    10800,
                    11750,
                ],
                "VT Controlsphere Viscose Easier": [
                    1100,
                    1600,
                    2100,
                    2450,
                    2750,
                    3175,
                    3500,
                    3800,
                    4100,
                ],
                "Air Angelic 4 Voltaic Easy 70%": [
                    2600,
                    3050,
                    3575,
                    3950,
                    4175,
                    4475,
                    4700,
                    4850,
                    5100,
                ],
                "cloverRawControl Viscose Easier 50cm": [
                    5000,
                    5850,
                    6400,
                    7100,
                    7800,
                    9200,
                    10000,
                    10700,
                    11500,
                ],
                "Flower Easier 50cm": [1050, 1350, 1650, 2000, 2350, 2575, 2875, 3200, 3475],
                "PGTI Voltaic Easy Smoother": [550, 700, 1050, 1200, 1375, 1675, 1950, 2250, 2500],
                "Air CELESTIAL No UFO Easier": [800, 817, 842, 852, 862, 873, 883, 889, 896],
                "RawControlSphere Easier": [
                    4700,
                    5900,
                    7400,
                    8200,
                    8900,
                    10400,
                    11000,
                    12300,
                    13200,
                ],
                "Ground Plaza Sparky v3 Easy": [780, 805, 833, 845, 854, 866, 876, 884, 889],
                "Air Spectral Easy 85%": [870, 895, 907, 913, 917, 922, 926, 929, 932],
                "VT Aether Bot 1 Easier": [1200, 1600, 1850, 2100, 2400, 2700, 3000, 3250, 3475],
                "Plink Palace Easy Less Blinks": [
                    1300,
                    1600,
                    2000,
                    2175,
                    2300,
                    2525,
                    2775,
                    2925,
                    3100,
                ],
                "Air Pure Easier No UFO": [875, 895, 912, 919, 923, 928, 933, 937, 940],
                "Air Voltaic Invincible 4 Easier": [
                    1750,
                    2350,
                    2800,
                    3050,
                    3200,
                    3500,
                    3900,
                    4100,
                    4250,
                ],
                "ww5t Voltaic Slightly Larger": [70, 80, 92, 97, 101, 108, 113, 119, 125],
                "voxTargetSwitch 2 Large": [59, 68, 78, 84, 92, 99, 106, 113, 120],
                "aimerz+ Static Switching 6 Bot Slightly Larger": [
                    78,
                    89,
                    98,
                    103,
                    109,
                    115,
                    121,
                    126,
                    131,
                ],
                "1w6ts reload v2 20% bigger": [64, 71, 78, 85, 91, 97, 103, 109, 115],
                "beanTS Larger": [53, 64, 75, 86, 96, 106, 116, 126, 137],
                "StaticSwitchingVox": [67, 76, 86, 91, 96, 101, 108, 114, 119],
                "1w2ts reload smallflicks larger": [76, 85, 93, 100, 106, 112, 118, 124, 130],
                "eth Pasu Micro Easier": [600, 700, 775, 850, 925, 1000, 1075, 1150, 1225],
                "waldoTS Novice": [51, 63, 74, 85, 96, 107, 118, 129, 140],
                "kinTS Voltaic Easy 85%": [35, 38, 44, 49, 54, 59, 63, 67, 72],
                "domiSwitch Easier": [3000, 3600, 4200, 4800, 5400, 6000, 6500, 7000, 7500],
                "B180T Voltaic Easy Slower": [32, 37, 42, 47, 52, 57, 62, 67, 72],
                "Pasu Voltaic Reload Easier": [38, 46, 52, 59, 65, 72, 78, 84, 91],
                "VT Bounceshot Viscose Easier": [300, 380, 460, 540, 610, 690, 750, 810, 870],
                "CatClick Easier": [27, 35, 42, 49, 56, 63, 69, 75, 80],
                "1w3ts Pasu Perfected Micro Goated Larger 80%": [
                    420,
                    530,
                    640,
                    750,
                    860,
                    970,
                    1080,
                    1190,
                    1300,
                ],
                "Popcorn MV Easier": [90, 150, 200, 240, 280, 320, 360, 390, 420],
                "skyClick Goated Easier": [60, 71, 80, 89, 95, 102, 107, 112, 117],
                "VoxTS Click Easier": [48, 57, 64, 72, 79, 86, 93, 99, 105],
                "VT Floating Heads Viscose Easier": [400, 500, 580, 645, 700, 790, 870, 930, 1015],
                "psalmTS Viscose Click Easier": [31, 39, 45, 51, 57, 63, 69, 74, 80],
            },
            "Medium (9 ranks Cinnabar->Fuchsia)": {
                "Smoothsphere Viscose": [
                    6000,
                    7700,
                    8900,
                    10100,
                    11200,
                    12050,
                    12800,
                    13700,
                    14400,
                ],
                "Whisphere Viscose": [8600, 11000, 13000, 14750, 16750, 18250, 19750, 21350, 22750],
                "SmoothBot Perfected": [2450, 3225, 3725, 4200, 4600, 4950, 5300, 5650, 5950],
                "Leapstrafes Control wobin": [1750, 2200, 2500, 2850, 3200, 3425, 3700, 4000, 4300],
                "Controlsphere SuperbAim Viscose": [
                    6500,
                    8050,
                    9150,
                    10000,
                    11000,
                    11900,
                    12750,
                    13600,
                    14500,
                ],
                "VT Controlsphere Viscose": [1450, 1975, 2350, 2725, 3075, 3400, 3700, 4050, 4390],
                "Air Angelic 4 Voltaic Easy 90%": [
                    2375,
                    2825,
                    3150,
                    3450,
                    3700,
                    3925,
                    4125,
                    4360,
                    4550,
                ],
                "cloverRawControl Viscose 50cm": [
                    4950,
                    6250,
                    7350,
                    8500,
                    9500,
                    10400,
                    11300,
                    12100,
                    13050,
                ],
                "Flower 50cm": [1400, 1875, 2200, 2500, 2825, 3175, 3500, 3900, 4275],
                "VT PreciseOrb Intermediate 15% Slower": [
                    1000,
                    1400,
                    1700,
                    1975,
                    2225,
                    2500,
                    2725,
                    3000,
                    3250,
                ],
                "Air CELESTIAL No UFO Medium": [817, 847, 863, 877, 886, 893, 900, 906, 911],
                "RawControlSphere": [6600, 8150, 9300, 10500, 11300, 12100, 13100, 14075, 14900],
                "Ground Plaza Sparky V3": [850, 860, 870, 879, 886, 892.5, 899, 904.5, 909],
                "Air Spectral Easy": [890, 903, 910, 916, 921, 925, 929, 932.5, 935],
                "VT Aether Novice S5 Hard Bot 1 90%": [
                    1575,
                    2150,
                    2500,
                    2800,
                    3075,
                    3350,
                    3575,
                    3825,
                    4025,
                ],
                "Plink Palace Easy": [1825, 2250, 2500, 2675, 2875, 3025, 3250, 3450, 3625],
                "Air Pure Medium": [870, 888, 898, 904, 910.5, 915.5, 921, 926, 930],
                "Air Voltaic Invincible 4 Medium": [
                    2275,
                    2700,
                    3025,
                    3300,
                    3575,
                    3850,
                    4075,
                    4375,
                    4600,
                ],
                "ww4t Voltaic": [90, 102, 112, 118, 126, 133, 141, 152, 160],
                "voxTargetSwitch 2": [69, 78, 87, 96, 105, 115, 122, 129, 135],
                "aimerz+ Static Switching 5 Bot Slightly Larger": [
                    101,
                    111,
                    120,
                    126,
                    133,
                    139,
                    145,
                    153,
                    161,
                ],
                "1w6ts reload v2": [70, 82, 94, 105, 116, 123, 130, 138, 146],
                "beanTS": [80, 91, 102, 113, 125, 134, 142, 148, 155],
                "StaticSwitchingVox Small": [91, 101, 111, 119, 126, 133, 138, 142, 147],
                "1w2ts reload smallflicks slightly larger": [
                    99,
                    109,
                    119,
                    127,
                    135,
                    142,
                    150,
                    161,
                    169,
                ],
                "eth Pasu Micro Medium": [850, 975, 1050, 1150, 1250, 1300, 1400, 1525, 1650],
                "waldoTS Intermediate": [92, 104, 115, 126, 136, 144, 154, 163, 171],
                "kinTS Voltaic Easy": [42, 49, 55, 61, 67, 72, 76, 81, 86],
                "domiSwitch Easy": [3600, 4200, 4700, 5200, 5700, 6200, 6700, 7200, 7700],
                "B180T Voltaic Easy": [40, 49, 56, 63, 69, 75, 81, 86, 91],
                "Pasu Voltaic Reload Easy": [51, 64, 72, 80, 88, 96, 103, 112, 121],
                "VT Bounceshot Viscose Medium": [420, 540, 600, 690, 770, 850, 920, 1010, 1100],
                "CatClick Medium": [39, 49, 56, 63, 70, 77, 84, 95, 103],
                "1w3ts Pasu Perfected Micro Goated Larger": [
                    790,
                    850,
                    975,
                    1075,
                    1175,
                    1275,
                    1400,
                    1550,
                    1650,
                ],
                "Popcorn MV Medium": [160, 220, 260, 310, 350, 390, 440, 490, 540],
                "skyClick Goated Medium": [83, 91, 100, 109, 117, 125, 132, 143, 152],
                "VoxTS Click Medium": [62, 74, 81, 89, 96, 103, 110, 119, 127],
                "VT Floating Heads Viscose Medium": [475, 575, 650, 750, 825, 900, 975, 1085, 1200],
                "psalmTS Viscose Click": [46, 56, 62, 69, 75, 81, 88, 96, 104],
            },
            "Hard (8 ranks Wool->Silk)": {
                "Smoothsphere Viscose Hard": [9000, 9700, 10350, 11050, 11625, 12050, 12850, 13400],
                "Whisphere Viscose Hard": [14750, 15850, 16750, 18000, 18750, 19750, 21250, 22250],
                "SmoothBot Perfected Hard": [4050, 4325, 4550, 4800, 5000, 5350, 5550, 5725],
                "Leapstrafes Control wobin Hard": [2875, 3100, 3275, 3525, 3675, 3875, 4075, 4225],
                "Controlsphere SuperbAim Viscose Hard": [
                    9650,
                    10450,
                    11000,
                    11600,
                    12200,
                    13000,
                    13500,
                    14000,
                ],
                "VT Controlsphere Viscose Hard": [2700, 2975, 3225, 3450, 3625, 3875, 4200, 4475],
                "Air Angelic 4 Voltaic 90%": [3375, 3635, 3825, 4025, 4180, 4375, 4540, 4660],
                "cloverRawControl Viscose Hard 50cm": [
                    7800,
                    8375,
                    8950,
                    9600,
                    10400,
                    11100,
                    11650,
                    12200,
                ],
                "Flower Hard 50cm": [2650, 2950, 3100, 3400, 3600, 3925, 4125, 4325],
                "PGTI Voltaic Slower - 80%": [1800, 2100, 2375, 2650, 2900, 3150, 3400, 3625],
                "Air CELESTIAL Harder": [853, 865, 871, 879, 884, 889, 894, 897.5],
                "RawControlSphere Hard": [10775, 11400, 12000, 12650, 13300, 13950, 14650, 15200],
                "Ground Plaza Sparky V3 Viscose Hard": [
                    889,
                    892.5,
                    896.5,
                    900,
                    903,
                    906.5,
                    909,
                    911.5,
                ],
                "Air Spectral": [898, 909, 913, 917, 921, 924, 927, 929.5],
                "VT Aether Intermediate S5 Hard Bot 1": [
                    2300,
                    2550,
                    2800,
                    3050,
                    3235,
                    3375,
                    3565,
                    3725,
                ],
                "Plink Palace": [2775, 2925, 3025, 3175, 3325, 3435, 3510, 3600],
                "Air Pure Hard": [892, 897, 901, 905, 908, 912, 915, 916.5],
                "Air Voltaic Invincible 4 Slightly Slower": [
                    3125,
                    3325,
                    3475,
                    3675,
                    3850,
                    4025,
                    4175,
                    4300,
                ],
                "ww3t Voltaic": [131, 138, 144, 150, 157, 163, 170, 175],
                "voxTargetSwitch 2 10% Smaller": [109, 115, 119, 123, 126, 130, 135, 140],
                "aimerz+ Static Switching 5 Bot": [131, 136, 142, 146, 150, 154, 158, 162],
                "1w5ts reload": [117, 123, 130, 137, 144, 149, 156, 162],
                "beanTS 200%": [123, 128, 134, 139, 144, 149, 155, 159],
                "StaticSwitchingVox xSmall": [110, 118, 124, 130, 136, 142, 146, 151],
                "1w2ts reload smallflicks": [145, 151, 157, 163, 168, 173, 179, 184],
                "eth Pasu Micro Hard": [1260, 1350, 1420, 1500, 1575, 1650, 1750, 1800],
                "waldoTS": [135, 142, 149, 155, 161, 167, 174, 178],
                "kinTS Voltaic": [65, 69, 74, 79, 83, 87, 90, 93],
                "domiSwitch": [5100, 5500, 5900, 6400, 6700, 7000, 7200, 7400],
                "B180T Voltaic": [65, 70, 75, 80, 85, 90, 94, 98],
                "Pasu Voltaic Reload": [81, 88, 94, 100, 104, 109, 116, 122],
                "VT Bounceshot Viscose Hard": [720, 800, 850, 910, 960, 1020, 1100, 1190],
                "CatClick": [70, 76, 81, 86, 91, 99, 107, 112],
                "1w3ts Pasu Perfected Micro Goated": [
                    980,
                    1060,
                    1140,
                    1230,
                    1320,
                    1420,
                    1510,
                    1580,
                ],
                "Popcorn MV Hard": [340, 380, 410, 440, 480, 510, 560, 610],
                "skyClick Goated Hard": [119, 127, 132, 140, 144, 148, 156, 164],
                "VoxTS Click Medium 10% Smaller": [101, 107, 113, 118, 122, 127, 135, 142],
                "VT Floating Heads Viscose Hard": [780, 875, 930, 1000, 1080, 1150, 1200, 1260],
                "psalmTS Viscose Click Hard": [72, 79, 85, 88, 93, 99, 108, 114],
            },
            "Expert (6 ranks Interloper->Eclipse)": {
                "Smoothsphere Viscose Expert": [7500, 8500, 9500, 10500, 11500, 12500],
                "Whisphere Viscose Expert": [14500, 16000, 17500, 19000, 20500, 22000],
                "SmoothBot Perfected Expert": [4150, 4400, 4650, 4900, 5150, 5400],
                "Leapstrafes Control wobin Expert": [2900, 3200, 3450, 3700, 3950, 4200],
                "Controlsphere SuperbAim Viscose Expert": [9800, 10500, 11300, 12100, 12800, 13500],
                "VT Controlsphere Viscose Expert": [3150, 3400, 3650, 3900, 4150, 4400],
                "Air Angelic 4 Voltaic Slightly Smaller": [3200, 3450, 3650, 3850, 4050, 4250],
                "cloverRawControl Viscose Expert 50cm": [7200, 8000, 8800, 9500, 10200, 10900],
                "Flower Expert 50cm": [3300, 3550, 3750, 3950, 4150, 4350],
                "PGTI Voltaic": [2000, 2200, 2400, 2600, 2800, 3000],
                "Air CELESTIAL Expert": [881, 885, 889, 893, 896, 899],
                "RawControlSphere Expert": [9300, 10300, 11300, 12300, 13200, 14100],
                "Ground Plaza Sparky V3 Thin": [890, 894, 897, 900, 903, 906],
                "Air Spectral Harder": [914, 918, 922, 925, 927, 929],
                "VT Aether Advanced S5 bot 1": [2700, 2900, 3100, 3250, 3400, 3550],
                "Plink Palace Hard": [2750, 2950, 3125, 3300, 3475, 3650],
                "Air Pure": [899, 902, 905, 908, 911, 914],
                "Air Voltaic Invincible 4": [3650, 3800, 3950, 4100, 4200, 4300],
                "ww3t Voltaic 10% Smaller": [152, 158, 163, 168, 173, 178],
                "voxTargetSwitch 2 20% Smaller": [122, 126, 130, 133, 136, 139],
                "aimerz+ Static Switching": [139, 144, 148, 153, 158, 163],
                "1w4ts Voltaic": [138, 144, 150, 156, 161, 167],
                "beanTS 200% Slightly Smaller": [132, 137, 142, 146, 150, 155],
                "StaticSwitchingVox xxSmall": [128, 132, 136, 140, 144, 148],
                "1w2ts reload smallflicks 25% smaller": [146, 152, 158, 163, 168, 173],
                "eth Pasu Micro Expert": [1325, 1400, 1475, 1550, 1625, 1700],
                "waldoTS Elite": [146, 152, 157, 162, 167, 172],
                "kinTS Voltaic Elite": [72, 77, 81, 85, 89, 93],
                "domiSwitch Harder": [5300, 5700, 6100, 6500, 6900, 7300],
                "B180T Voltaic 15% Smaller": [72, 77, 81, 85, 89, 93],
                "Pasu Voltaic Reload 15% Smaller": [95, 103, 108, 113, 118, 123],
                "VT Bounceshot Viscose Expert": [820, 900, 970, 1040, 1100, 1150],
                "CatClick Harder": [70, 78, 85, 92, 98, 104],
                "1w3ts Pasu Perfected Micro Goated 10% small": [1225, 1300, 1375, 1450, 1500, 1550],
                "Popcorn MV Advanced": [400, 430, 460, 490, 520, 550],
                "skyClick Goated Expert": [114, 121, 128, 135, 142, 149],
                "VoxTS Click Hard": [110, 118, 125, 132, 139, 146],
                "VT Floating Heads Viscose Expert": [880, 940, 1000, 1060, 1110, 1180],
                "psalmTS Viscose Click Expert": [76, 85, 93, 101, 109, 115],
            },
        },
        "accuracy_caveat": (
            "High confidence transcription (KovaaK's API ids 2335-2338 + official S2 sheet, "
            "fetched 2026-07-28). Targets were still being rebalanced through May-June 2026 "
            "(changelog: 'rebalanced difficulty of hare, ermine, puffin and penguin'; seal data "
            "dated 2026-06-03) — refresh from the KovaaK's backend endpoint before hard-coding. "
            "Validity rules carry over: 103 hfov minimum; Fingertip scenarios invalid slower than "
            "50cm/360. The 'Entry' tab was not extracted."
        ),
    },
    {
        "system": "Aimer7 guide skill tiers and score goals",
        "version": (
            "2019 (Jan 13, 2019 PDF; FOV 103 OW baseline; historical scenario versions — scores "
            "not comparable to VT-prefixed remakes)"
        ),
        "categories": {
            "structure": (
                "Seven progressive levels: complete beginners -> intermediate beginners -> "
                "advanced beginners -> sub-intermediate -> intermediate -> advanced -> 'sub aim "
                "beasts'; each level has 3 parallel routines (tracking-dominated, "
                "click-timing-dominated, complete). Advancement is time-in-level (10-15 "
                "consecutive days at beginner levels, 3-5 weeks at sub-intermediate) plus score "
                "gates."
            ),
        },
        "scenarios": {
            "governor": (
                "Universal rule on click-timing scenarios: 85% accuracy floor / 95% ceiling, "
                "whole-run, no accu-whoring."
            ),
        },
        "thresholds": {
            "Intermediate gates": {
                "Close Fast Strafes Invincible": "42%+ avg accuracy whole-run",
                "Vertical Fast Strafes": "38-40% accuracy",
                "Air": "99820+",
                "PatTargetSwitch": "6500+",
                "Bounce 180 Tracking": "75+",
                "1wall5targets pasu": "80+",
                "Pressure Aiming 7t": "12000+",
                "1w6ts small": "1100+",
            },
            "Accomplished advanced (tracking)": {
                "Close Fast Strafes Invincible": "11000+",
                "Vertical Fast Strafes": "10500+",
                "Ground Plaza": "99875+",
                "Air": "99835+",
                "PatTargetSwitch": "7200+",
                "Target Switching 360": "16000+",
                "pasu": "90+",
            },
            "Accomplished advanced (click-timing)": {
                "POPCORN": "1500+",
                "McCoy 1v1": "3300+",
                "Target Acquisition Flick": "90+",
                "1wall9000targets": "250+",
                "1w6ts TE": "185+",
            },
            "Sub aim beast": {
                "Air": "99845+",
                "Ground Plaza": "99883+",
                "Close Long Strafes Invincible": "18000+ (~78% acc)",
                "Close Fast Strafes Invincible": "12000+ (~50% acc)",
                "Vertical Fast Strafes": "55% = 'god' level",
                "PatTargetSwitch": "7500+ consistent",
                "POPCORN": "2000+",
                "McCoy": "3600+",
                "pasu": "105+",
            },
        },
        "accuracy_caveat": (
            "High confidence for what the guide says; LOW comparability today — 2019 scenario "
            "versions and scoring differ from current VT remakes, and KovaaK's scoring has "
            "changed. Use these as historical calibration and for the accuracy-percentage "
            "landmarks (which transfer better than scores), not as live targets."
        ),
    },
    {
        "system": "Research reference points (Donovan et al. 2022 / Listman et al. 2021)",
        "version": (
            "Aimlabs Gridshot (speed-emphasis) and Sixshot (precision-emphasis); N=32 pro/semi-pro "
            "(Valorant/PUBG/R6) and N=7,174 general players / 682,564 runs"
        ),
        "categories": {
            "expertise_markers": [
                "faster reaction times (p<1e-4 Gridshot)",
                "higher movement speed",
                "lower endpoint variability",
                (
                    "strategic swipiness modulation (~0.5 swipe on speed tasks, >=1 flick-and-land "
                    "on precision tasks; pros fire later on precision, earlier on speed)"
                ),
            ],
        },
        "scenarios": {
            "longitudinal": (
                "Hit rate improves modestly with practice; hits/second improves considerably; "
                "40-60% day-to-day retention of gains."
            ),
        },
        "thresholds": {
            "note": (
                "Directional markers only — the studies provide no population-normed cutoffs "
                "usable as rank thresholds."
            ),
        },
        "accuracy_caveat": (
            "High confidence for the findings; these are DIRECTIONAL research anchors, not "
            "benchmark ladders. Do not present study means as targets."
        ),
    },
]

ROUTINES: list[dict] = [
    {
        "id": "r-voltaic-fundamental",
        "name": "Voltaic Fundamental Routines (rank-gated, Bronze-Master)",
        "structure": (
            "60-minute daily complete routine per rank tier, mixing ~5-min smoothness-tracking "
            "blocks, 10-min clicking blocks, and 10-min switching blocks, each with a focus cue "
            "('as smooth as possible', 'switch as fast as possible', 'do not predict — be as "
            "reactive as possible'). Scenario families progress from long-strafe smoothness at "
            "Bronze (Thin aiming long slow, Air/GP far+close long strafes, 1wall2targets TE, "
            "Bounce 90T easy, patTargetSwitch 90 easy) to small/invincible variants at Master "
            "(B180TI Small Sparky, Smoothbot Unvincible Small Goated, patCircleSwitch small NR). "
            "Advancement gated on completing the next rank's benchmarks. Deliberately no score "
            "targets: routines train, benchmarks measure. Grandmaster+ build their own from the "
            "recommended-scenarios sheet. Alternate similar scenarios to avoid pattern "
            "memorization."
        ),
        "confidence": "high",
        "source": (
            "https://docs.google.com/document/d/1BPiDxbaqQVxInKwFfdr_AJdJDYVGKbUBGHJDKOeUy-Q "
            "(Voltaic fundamental routines, by sini)"
        ),
    },
    {
        "id": "r-vdim",
        "name": "Voltaic Daily Improvement Method (VDIM, by LG56)",
        "structure": (
            "Weekly periodization: each day of the week isolates one aiming subcategory, with "
            "playlists that ramp difficulty within the day up to that subcategory's benchmark "
            "scenario. A concrete community implementation of rotating subskills across days "
            "(contextual-interference + specificity hybrid) instead of training everything daily."
        ),
        "confidence": "medium",
        "source": (
            "https://steamcommunity.com/sharedfiles/filedetails/?id=3438743557 (VDIM S3 playlists)"
        ),
    },
    {
        "id": "r-aimer7-core",
        "name": "Aimer7 routine template (all levels)",
        "structure": (
            "~6 scenarios per session at 10-15 minutes each (60-90 min total); three parallel "
            "routine classes per level (tracking-dominated, click-timing-dominated, complete) — "
            "pick by main game; starred scenarios optional; parenthesized alternatives can "
            "substitute or split a slot. Advance levels by consecutive days trained (10-15 days at "
            "beginner levels, 3-5 weeks at sub-intermediate), not raw hours; cap useful volume at "
            "~2-3 h/day and split it if training seriously. At advanced level drop fixed times: "
            "play 1-2 scenarios per session (or per week) and rotate. Governor throughout: 85-95% "
            "accuracy band on click-timing scenarios; tracking accuracy goals counted whole-run "
            "only."
        ),
        "confidence": "high",
        "source": (
            "https://www.docdroid.net/file/download/ENPQRI5/training-1-pdf.pdf (Aimer7 2019, secs "
            "2-3.6)"
        ),
    },
    {
        "id": "r-aimer7-smoothness",
        "name": "Smoothness/precision protocol (for jitter/submovement flags)",
        "structure": (
            "For ~2 weeks: raise sensitivity 10-20% (e.g. 30cm/360 -> 24cm/360; a +50% variant "
            "exists), set FOV to 80 OW, use a small dot crosshair, and play thin/small tracking "
            "scenarios 20-30 min/day. Rationale: low FOV and a small dot make your motion visible "
            "so lack of smoothness is exposed; higher sens forces smoothness because it is harder "
            "to control. Then return to normal settings."
        ),
        "confidence": "high",
        "source": (
            "https://docs.google.com/document/d/1d1FY1qbwrgdj2K1wmhktbdcgG0ybeK3YJSeZ49yraW4 + "
            "https://steamcommunity.com/sharedfiles/filedetails/?id=1818885969 (Aimer7/Voltaic)"
        ),
    },
    {
        "id": "r-aimer7-speed",
        "name": "Speed protocol (for slow-but-accurate flags)",
        "structure": (
            "Divide sensitivity by 2-4 and play at 130 OW FOV on flick scenarios (Tile Frenzy "
            "variants, patTargetSwitch), holding fire for the whole run. Forces large arm "
            "movements: '100-degree flicks at 3x slower sens make you much better at 300-degree "
            "flicks on your original sens.' Static grid scenarios are legitimate ONLY in this "
            "speed-tool role."
        ),
        "confidence": "high",
        "source": (
            "https://docs.google.com/document/d/1d1FY1qbwrgdj2K1wmhktbdcgG0ybeK3YJSeZ49yraW4 + "
            "https://steamcommunity.com/sharedfiles/filedetails/?id=1818885969 (Aimer7/Voltaic)"
        ),
    },
    {
        "id": "r-aimer7-large-angle",
        "name": "Large-angle protocol (for off-center region deficits)",
        "structure": (
            "10 days to 2 weeks of 360-degree scenarios (Tile Frenzy 360 Strafing 400%, Target "
            "Switching 360, LG pin practice 360, etc.) at 103-130 OW FOV, trying never to lift the "
            "mouse during a run. Targets the fact that aim degrades with distance from the "
            "mousepad rest position."
        ),
        "confidence": "high",
        "source": (
            "https://docs.google.com/document/d/1d1FY1qbwrgdj2K1wmhktbdcgG0ybeK3YJSeZ49yraW4 "
            "(Large Angles section)"
        ),
    },
    {
        "id": "r-voltaic-reactivity",
        "name": "Reactivity protocol (intermediate+ only)",
        "structure": (
            "25-27 cm/360, 103 OW FOV, smallest visible dot crosshair — optionally disable the "
            "crosshair entirely to force visual acuity and reading. Thin fast-strafe scenarios, "
            "focusing on the bot and reading its movement, not the crosshair. VT Air cue: do not "
            "overreact to direction changes. If too hard, timescale scenarios down and fix "
            "technique first. Not for beginners — smoothness fundamentals come first."
        ),
        "confidence": "high",
        "source": (
            "https://docs.google.com/document/d/1d1FY1qbwrgdj2K1wmhktbdcgG0ybeK3YJSeZ49yraW4 + "
            "https://docs.google.com/document/d/1vHiQRZMBJlmI69-SgHm3i0eS5ALfq2hEu-ZPyvC6ycE"
        ),
    },
    {
        "id": "r-weekly-benchmark",
        "name": "Benchmark cadence",
        "structure": (
            "Run the benchmark set about once per week to measure; spend all other sessions on "
            "routines. Never grind benchmark scenarios as practice — they are tuned for scoring "
            "consistency, not for improvement."
        ),
        "confidence": "high",
        "source": (
            "https://docs.google.com/document/d/1BPiDxbaqQVxInKwFfdr_AJdJDYVGKbUBGHJDKOeUy-Q "
            "(Voltaic)"
        ),
    },
    {
        "id": "r-tenz-warmup",
        "name": "TenZ warmup (focused-warmup archetype)",
        "structure": (
            "Short Aim Lab playlist (Gridshot, Strafetrack, Motiontrack) -> Range work (flicking, "
            "strafing, drone shooting) -> minimum 20-30 min deathmatch, with the whole session "
            "focused on ONE specific aspect of aim (his stated preference, from the Aimlabs "
            "course). His self-designed tasks (VCT NA — ends if 3 targets pass you, reference "
            "61,230 pts @ 89.57% acc; 180s; Haven Clutch) exist to simulate panic and train "
            "composure. Caveat: his sens changes constantly — treat any TenZ number as 'most "
            "recently confirmed'."
        ),
        "confidence": "high (course statement), medium (full recipe aggregation)",
        "source": (
            "https://aimlabs.com/courses/7qomjBJ1VziOMBejmvHQ0a/lessons/48OoScMyHoZKBNlQfzfmQ9 + "
            "https://www.oneesports.gg/valorant/valorant-warm-up-routine-tenz-aim-lab/ + "
            "https://gamezo.gg/valorant-warmup-routines/"
        ),
    },
    {
        "id": "r-demon1",
        "name": "Demon1 routine (trainer-first archetype)",
        "structure": (
            "Daily Aimlabs-first routine opening with Speed Switch (precise snaps in quick "
            "succession, doubling as wrist warmup), then blocks mixing speed, tracking, accuracy, "
            "and precision; warms up with multiple weapons. Philosophy: precision over volume "
            "('you don't want to commit to a spray'), training should be enjoyable, and 'true "
            "development comes in the game.' Community KovaaK's port: Steam Workshop 'Improved "
            "DEMON1 Routine' (id 3039378337). Settings context: one of pro Valorant's lower senses "
            "(~196 eDPI, earlier 160), large arm movements."
        ),
        "confidence": "medium",
        "source": (
            "https://aimlabs.com/articles/valorant/demon1s-ultimate-aim-training-routine/ + "
            "https://sportskeeda.com/valorant/demon1-aim-routine-2024-how-aim-like-valorant-pro"
        ),
    },
    {
        "id": "r-scream",
        "name": "ScreaM warmup (stretch + micro-breaks archetype)",
        "structure": (
            "Wrist/hand stretches first -> ~20 min Aim Lab (Gridshot, Headshot, Switchtrack, "
            "Wallpeak variants) -> Range work while moving (crosshair placement, peeking) -> 1-2 "
            "optional deathmatches. Recommends 5-minute breaks between games — a pro-endorsed "
            "micro-rest pattern kovadapt can cite when fatigue trends worsen."
        ),
        "confidence": "medium",
        "source": "https://gamezo.gg/valorant-warmup-routines/",
    },
    {
        "id": "r-ethos",
        "name": "NRG Ethos warmup (weakness-first archetype)",
        "structure": (
            "30-40 min custom Aim Lab playlist (Sixshot, Motionshot, Switchtrack) -> Range: "
            "counter-strafing, tapping, flicking, spraying with an accuracy-before-speed rule -> "
            "deathmatch with per-weapon win goals (Phantom, Vandal, Operator). Explicit doctrine: "
            "identify weaknesses, then train them."
        ),
        "confidence": "medium",
        "source": "https://gamezo.gg/valorant-warmup-routines/",
    },
    {
        "id": "r-s1mple",
        "name": "s1mple daily structure (hybrid DM-heavy archetype)",
        "structure": (
            "~30 min individual warmup mixing modalities in one block (taps, short-burst sprays, "
            "fast flicks, frequent weapon switching) -> ~8 h team practice -> FPL in free time. "
            "Deathmatch even during tournaments; small 1v1 duel maps for AWP practice. Rest "
            "cadence: one day off per week, plus a 5-day break every ~2 months. Sens held "
            "effectively constant for years (3.09 @ 400 DPI)."
        ),
        "confidence": "high (interview quotes), medium (warmup content)",
        "source": (
            "https://plarium.com/en/blog/interview-s1mple/ + "
            "https://cs.money/blog/esports/how-s1mple-trains-maps-aim-awp/"
        ),
    },
    {
        "id": "r-donk",
        "name": "donk daily structure (in-game-volume counterpoint)",
        "structure": (
            "Wake 2 h before the noon team meeting -> ~2 h VOD/strategy -> from 14:00 at least 4 "
            "consecutive scrim maps -> 2-6 FACEIT games in the evening (~10+ competitive "
            "maps/day). Deathmatch only on felt need — he calls DM duels 'unrealistic'. ZywOo is "
            "the same pole: no individual practice at all, FACEIT plus surf/KZ for fun. Include as "
            "the honest counterweight: trainer routines are one valid path, not the only one."
        ),
        "confidence": "medium",
        "source": (
            "https://esports.gg/news/counter-strike-2/donk-reveals-his-cs2-practice-routine/ + "
            "https://pley.gg/cs2/zywoo-dont-individual-practice-only-surf-kz-playing-fun/"
        ),
    },
    {
        "id": "r-kovadapt-session",
        "name": "Synthesized kovadapt session template (editorial, from converging sources)",
        "structure": (
            "1) Warmup <= 15 min, one focus, set-reinstating not exhausting (Adams; Voltaic; "
            "TenZ). 2) Main block 30-60 min of routine work at the challenge point — kovadapt's "
            "adaptive variant holding accuracy in the archetype band (Aimer7 governor; Guadagnoli "
            "& Lee). 3) One weakness-specific block driven by the current top diagnostic "
            "(region/bias/smoothness/speed protocol). 4) End the session when the fatigue trend "
            "turns — gains consolidate overnight (Walker; Aimer7). 5) Benchmark once a week, judge "
            "progress by EWMAs and Fitts-slope trend, not PBs. Daily short sessions >= 5 days/week "
            "beat weekend marathons (Donovan & Radosevich)."
        ),
        "confidence": "synthesis (each element individually high-confidence)",
        "source": (
            "Synthesis of Adams 1961, Guadagnoli & Lee 2004, Walker 2002, Donovan & Radosevich "
            "1999, Aimer7 2019, Voltaic getting-started/fundamentals docs"
        ),
    },
]

GAPS: tuple[str, ...] = (
    (
        "No primary source defines numeric cutoffs for kovadapt's telemetry metrics (what "
        "overshoot_rate, corrections-per-flick, or Fitts-slope value counts as 'high' at a given "
        "rank). Donovan et al. give directional expertise markers only. All numeric conditions in "
        "the diagnostics (e.g. overshoot_rate > 0.3) are editorial calibrations that should be "
        "tuned against kovadapt's own collected data."
    ),
    (
        "Input-health norms are unsourced: no primary community document specifies acceptable "
        "polling-rate jitter (jitter_ms) or minimum polling for training validity; the 1000 Hz+ "
        "norm is inferred from pro settings trackers (secondary). The dx-input-health diagnostic "
        "is the weakest-sourced entry."
    ),
    (
        "No official mapping exists between KovaaK's scenario SCORES and accuracy/telemetry "
        "metrics: per-scenario scoring formulas (sqrt-accuracy, reload/ammo systems, MBS caps) are "
        "only partially documented, so kovadapt cannot convert its measured accuracy into "
        "predicted Voltaic energy."
    ),
    (
        "The Voltaic S5 official benchmark-rules document (FOV minimum, pause rules, VOD "
        "requirements for the current season) was not directly fetched; rules were corroborated "
        "indirectly from S3-era posts and the Viscose sheet (medium confidence)."
    ),
    (
        "Unresolved wording conflict noted (resolved by weight of evidence): the Voltaic S4 "
        "benchmarks doc says overall energy is 'the sum of all the subcategory energy values' "
        "while the S5 Instructions tab and app.voltaic.gg describe a harmonic-mean-style average "
        "with lower scores weighted more heavily. Knowledge base uses harmonic mean; the S4-era "
        "aggregation detail may genuinely have differed."
    ),
    (
        "The claim that right-handed players are usually weaker tracking rightward is anecdotal "
        "(Aimer7/Voltaic issue doc); no research quantifying directional aim asymmetry was found. "
        "The prescription (bias practice toward the weak side) is primary, the prevalence claim is "
        "not."
    ),
    (
        "Aimer7's target-focus vs crosshair-focus gaze doctrine is labeled a theory by its own "
        "author; no eye-tracking study validating it was found."
    ),
    (
        "Sensitivity-stability doctrine remains genuinely contested: Voltaic endorses changing "
        "sens (including randomizers), Aimer7 forbids changing settings to inflate scores but "
        "prescribes temporary changes for training, and pro practice spans both poles (s1mple "
        "stable for years, TenZ constantly tweaking). Encoded as a spectrum, not resolved."
    ),
    (
        "Voltaic S3 per-scenario thresholds were not transcribed (structure and rank names only); "
        "the Viscose S2 'Entry' tab was not extracted; the 'Voltaic S5.5' revision (KovaaK's id "
        "2070) has no official announcement — its status and permanence are unknown."
    ),
    (
        "Viscose S2 thresholds were still being rebalanced through June 2026; hard-coded values "
        "may already be stale. kovadapt should refresh from the KovaaK's backend benchmark "
        "endpoint (documented in the benchmarks entries) rather than trusting static numbers."
    ),
    (
        "NiKo's widely-cited 3-4 h/day deathmatch advice could not be verified against its "
        "purported source; treated as folklore and excluded from routines."
    ),
    (
        "s1mple, donk, and ZywOo routine details come from geo-blocked or secondary compilations "
        "(cs.money, esports.gg, pley.gg summaries of interviews); direct-quote confidence is high "
        "only for the Plarium s1mple interview and the ZywOo Dust2 interview quote."
    ),
    (
        "No source quantifies a within-session time threshold at which aim performance measurably "
        "degrades (Aimer7's 2-3 h/day cap is experiential); kovadapt's Theil-Sen fatigue detection "
        "is the measurement, doctrine only supplies the response (ease, then stop)."
    ),
    (
        "No community or research source provides expected values for corrective-submovement "
        "counts per archetype at each rank tier (e.g. 'Gold players average 1.4 corrections on "
        "switches') — the zero-correction switching ideal and one-clean-correction static ideal "
        "are endpoints, not a graded ladder."
    ),
)


def principle(pid: str) -> dict:
    """Return the PRINCIPLES entry for ``pid``. Raises KeyError on unknown ids."""
    return PRINCIPLES[pid]


def diagnostic(did: str) -> dict:
    """Return the DIAGNOSTICS entry for ``did``. Raises KeyError on unknown ids."""
    return DIAGNOSTICS[did]


def sources_for(entry_id: str) -> tuple:
    """Return the sources tuple for a principle or diagnostic id. Raises KeyError on unknown."""
    if entry_id in PRINCIPLES:
        return PRINCIPLES[entry_id]["sources"]
    if entry_id in DIAGNOSTICS:
        return DIAGNOSTICS[entry_id]["sources"]
    raise KeyError(entry_id)
