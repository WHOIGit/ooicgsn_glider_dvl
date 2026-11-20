"""Build the DVL processing pipeline page: an overview and one flowchart per notebook."""

from html import escape
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------- flowchart engine
W = 800  # viewBox width
SPINE_X, SPINE_W = 40, 360  # main column
CX = SPINE_X + SPINE_W / 2
RIGHT_X, RIGHT_W = 440, 330  # side column (branches and notes)
GAP = 36


def t(x, y, s, cls, anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>'


def proc_height(n):
    h = 14 + 17 * len(n.get("title", [])) + 16 * (1 if n.get("fn") else 0) + 15 * len(n.get("detail", []))
    return h + 8


def box(x, y, w, n, cls):
    """A process-like box with a title, a function name, and detail lines."""
    h = proc_height(n)
    out = [f'<rect x="{x}" y="{y:.1f}" width="{w}" height="{h:.1f}" rx="6" class="{cls}"/>']
    yy = y + 22
    for line in n.get("title", []):
        out.append(t(x + 14, yy, line, "tt"))
        yy += 17
    if n.get("fn"):
        out.append(t(x + 14, yy, n["fn"], "fn"))
        yy += 16
    for line in n.get("detail", []):
        out.append(t(x + 14, yy, line, "dt"))
        yy += 15
    return out, h


def data_box(x, y, w, n):
    lines = n["files"]
    h = 16 + 16 * len(lines) + (17 if n.get("title") else 0)
    out = [f'<rect x="{x}" y="{y:.1f}" width="{w}" height="{h:.1f}" rx="14" class="n-data"/>']
    yy = y + 22
    if n.get("title"):
        out.append(t(x + 16, yy, n["title"], "tt"))
        yy += 17
    for f in lines:
        out.append(t(x + 16, yy, f, "fl"))
        yy += 16
    return out, h


def notes(x, y, lines):
    out = []
    yy = y + 18
    for line in lines:
        out.append(t(x, yy, line, "nt"))
        yy += 15
    return out, (yy - y)


def arrow(x1, y1, x2, y2, cls="e"):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" marker-end="url(#ah)"/>'


def flowchart(nodes, label):
    out, y, prev_bottom = [], 16, None
    pending_label = None
    for n in nodes:
        kind = n["kind"]
        if kind == "group":
            y += 8
            out.append(f'<line x1="20" y1="{y:.1f}" x2="{W - 20}" y2="{y:.1f}" class="rule"/>')
            out.append(t(CX + 14, y + 20, n["text"].upper(), "gh"))
            y += 34
            continue
        if prev_bottom is not None:
            out.append(arrow(CX, prev_bottom, CX, y - 2))
            if pending_label:
                out.append(t(CX + 8, prev_bottom + 15, pending_label, "el"))
        pending_label = None
        if kind == "proc":
            parts, h = box(SPINE_X, y, SPINE_W, n, "n-proc")
            out += parts
            side_h = 0
            if n.get("notes"):
                nparts, side_h = notes(RIGHT_X, y, n["notes"])
                out += nparts
            bottom = y + h
            y = y + max(h, side_h) + GAP
        elif kind == "data":
            parts, h = data_box(SPINE_X, y, SPINE_W, n)
            out += parts
            side_h = 0
            if n.get("notes"):
                nparts, side_h = notes(RIGHT_X, y, n["notes"])
                out += nparts
            bottom = y + h
            y = y + max(h, side_h) + GAP
        elif kind == "dec":
            h = 74
            mid = y + h / 2
            hw = SPINE_W / 2
            pts = f"{CX},{y} {CX + hw},{mid} {CX},{y + h} {CX - hw},{mid}"
            out.append(f'<polygon points="{pts}" class="n-dec"/>')
            q = n["q"]
            y0 = mid + 4 - (len(q) - 1) * 8
            for i, line in enumerate(q):
                out.append(t(CX, y0 + i * 16, line, "dq", "middle"))
            side = n["side"]
            scls = "n-stop" if side.get("stop") else "n-side"
            parts, sh = box(RIGHT_X, mid - 20, RIGHT_W, side, scls)
            out += parts
            out.append(arrow(CX + hw, mid, RIGHT_X - 2, mid, "e"))
            out.append(t((CX + hw + RIGHT_X) / 2, mid - 6, n["right"], "el", "middle"))
            pending_label = n["down"]
            bottom = y + h
            y = max(y + h, mid - 20 + sh) + GAP
        prev_bottom = bottom
    H = y - GAP + 16
    return svg(out, W, H, label)


def svg(parts, w, h, label):
    defs = (
        '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ahd"/>'
        "</marker></defs>"
    )
    return (
        f'<svg viewBox="0 0 {w} {h:.0f}" role="img" aria-label="{escape(label)}" '
        f'class="chart" style="min-width:{min(w, 680)}px">{defs}{"".join(parts)}</svg>'
    )


# ---------------------------------------------------------------- overview (top to bottom)
def overview():
    out = []
    ow = 980
    cw, ch = 178, 62

    def nb(x, y, w, title, subs, cls="n-nb"):
        h = 34 + 17 * len(subs)
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>')
        out.append(t(x + w / 2, y + 25, title, "nbt", "middle"))
        for i, s in enumerate(subs):
            out.append(t(x + w / 2, y + 44 + 17 * i, s, "dt", "middle"))
        return y + h

    def chip(x, y, name, desc):
        lines = name.split("|")
        out.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="14" class="n-data"/>')
        y0 = y + (26 if len(lines) == 1 else 20)
        for i, line in enumerate(lines):
            out.append(t(x + cw / 2, y0 + 15 * i, line, "fl", "middle"))
        out.append(t(x + cw / 2, y + (47 if len(lines) == 1 else 53), desc, "dt", "middle"))

    xs = [20 + i * (cw + 12) for i in range(5)]
    mid = lambda x: x + cw / 2  # noqa: E731

    nb1_bottom = nb(330, 16, 320, "1 · Data access and download", ["OOINet M2M · THREDDS · IOOS GDAC ERDDAP"])
    raw_y = nb1_bottom + 56
    out.append(t(20, raw_y - 12, "DATA/RAW", "gh"))
    raw = [
        ("adcp_data.nc", "DVL water track"),
        ("adcp_bottom_track.nc", "DVL bottom track"),
        ("science.nc", "GDAC CTD, profiles"),
        ("eng_recovered.csv", "glider engineering"),
        ("mooring_locations.csv", "Pioneer moorings"),
    ]
    for x, (a, b) in zip(xs, raw):
        chip(x, raw_y, a, b)
        out.append(arrow(490, nb1_bottom, mid(x), raw_y - 2, "e-soft"))

    nb2_y = raw_y + ch + 56
    nb2_bottom = nb(330, nb2_y, 320, "2 · Process and merge", ["corrections, merge, profiles, links"])
    for x in xs[:4]:
        out.append(arrow(mid(x), raw_y + ch, 490, nb2_y - 2, "e-soft"))

    pro_y = nb2_bottom + 56
    out.append(t(20, pro_y - 12, "DATA/PROCESSED", "gh"))
    pro = [
        ("merged.nc", "ADCP times · dives"),
        ("glider.nc", "eng times · dives+climbs"),
        ("depth_averaged_|current.csv", "201 surfacings"),
        ("science_profiles.csv", "1385 dives, 1385 climbs"),
        ("profile_links.csv", "ADCP ↔ science"),
    ]
    for x, (a, b) in zip(xs, pro):
        chip(x, pro_y, a, b)
        out.append(arrow(490, nb2_bottom, mid(x), pro_y - 2, "e-soft"))

    nb4_y = pro_y + ch + 56
    nb(20, nb4_y, 300, "3 · 3.1 · Data analysis", ["exploratory: track maps, QC, simple bins"], "n-nb3")
    nb4_bottom = nb(
        400, nb4_y, 420, "4 · Remove the glider motion",
        ["flight model, bottom track, inversion, checks", "also reads the raw bottom track and eng data"],
    )
    out.append(arrow(mid(xs[0]), pro_y + ch, 170, nb4_y - 2, "e-soft"))
    for x in xs[:3]:
        out.append(arrow(mid(x), pro_y + ch, 610, nb4_y - 2, "e-soft"))

    out_y = nb4_bottom + 56
    out.append(t(20, out_y - 12, "DATA/PROCESSED", "gh"))
    outs = [
        ("velocity_|profiles_dac.nc", "Method 1"),
        ("velocity_profiles_|bottom_track.nc", "Method 2"),
        ("bottom_track_|profiles.nc", "BT-referenced, binned"),
        ("flight_calibration.csv", "Vg, Cd0, pitch offset"),
        ("depth_averaged_|current_check.csv", "per surfacing"),
    ]
    for x, (a, b) in zip(xs, outs):
        chip(x, out_y, a, b)
        out.append(arrow(610, nb4_bottom, mid(x), out_y - 2, "e-soft"))

    nx_y = out_y + ch + 50
    out.append(f'<rect x="330" y="{nx_y}" width="320" height="50" rx="8" class="n-next"/>')
    out.append(t(490, nx_y + 21, "Next · objective mapping of sections", "tt", "middle"))
    out.append(t(490, nx_y + 38, "not built yet", "dt", "middle"))
    out.append(arrow(mid(xs[0]), out_y + ch, 470, nx_y - 2, "e-dash"))
    out.append(arrow(mid(xs[1]), out_y + ch, 510, nx_y - 2, "e-dash"))
    return svg(
        out, ow, nx_y + 66,
        "Data flow between the notebooks: downloads, raw files, processing, processed files, "
        "profile construction, outputs",
    )


# ---------------------------------------------------------------- notebook 1
NB1 = [
    {"kind": "data", "title": "Inputs", "files": ["CP05MOAS ADCP datasets (M2M search)", "asset-management deployment sheets → FZ line"]},
    {"kind": "group", "text": "for each glider deployment"},
    {"kind": "proc", "title": ["Download the DVL streams"], "fn": "load_kdata(…, 'recovered_host', stream)",
     "detail": ["adcp_data, bottom_track, config, engineering"],
     "notes": ["load_kdata works only on the OOI JupyterHub"]},
    {"kind": "dec", "q": ["Any DVL stream", "with data?"], "down": "yes", "right": "none",
     "side": {"title": ["Log: missing"], "detail": ["some missing → partial"], "stop": True}},
    {"kind": "proc", "title": ["Download the engineering data"], "fn": "THREDDS catalog → CSV",
     "detail": ["NTP seconds → datetime"]},
    {"kind": "dec", "q": ["Recovered (host)", "data available?"], "down": "yes", "right": "no",
     "side": {"title": ["Use telemetered data"], "detail": ["log: partial, telemetered only"]}},
    {"kind": "proc", "title": ["Find and download the GDAC dataset"], "fn": "ERDDAP tabledap · cp_<glider>-<start>",
     "detail": ["start within 1 day of the deployment", "prefer the -delayed dataset", "time ← precise_time"]},
    {"kind": "dec", "q": ["Download", "succeeded?"], "down": "yes", "right": "no",
     "side": {"title": ["Log: failed"], "detail": ["error message kept, retried later"], "stop": True}},
    {"kind": "data", "title": "Saved to data/raw", "files": ["<refdes>.deployment<NNNN>.adcp_*.nc", "deployment<NNNN>_<eng>-recovered_host….csv", "<glider>.deployment<NNNN>.science.nc", "download_log.csv"]},
    {"kind": "proc", "title": ["Mooring locations"], "fn": "asset-management *_Deploy.csv",
     "detail": ["→ data/external/mooring_locations.csv"]},
]

# ---------------------------------------------------------------- notebook 2
NB2 = [
    {"kind": "data", "title": "Load (data/raw)", "files": ["adcp_data.nc", "adcp_bottom_track.nc", "science.nc (GDAC)"],
     "notes": ["604,054 DVL ensembles, 30 cells × 2 m", "GDAC: 2770 profiles, 3.8 M times"]},
    {"kind": "group", "text": "Corrections to the OOINet ADCP data"},
    {"kind": "dec", "q": ["Attitude > 360° / > 90°?", "(hundredths of a degree)"], "down": "no",
     "right": "yes", "side": {"title": ["Divide heading, pitch, roll", "by 100"], "fn": "adcp_attitude_to_degrees"}},
    {"kind": "dec", "q": ["OOINet velocities already", "kept as ooi_*?"], "down": "yes: skip",
     "right": "no", "side": {"title": ["Rebuild east/north velocities"], "fn": "true_north_velocities",
                              "detail": ["= water_velocity_east/north ÷ 1000", "OOINet's −14.21° rotation kept as ooi_*", "(heading is already true north)"]}},
    {"kind": "proc", "title": ["Estimate the DVL clock offset"], "fn": "estimate_time_offset",
     "detail": ["per dive: bottom range ~ CTD depth(t + lag) + time", "trial lags ±15 s in 0.25 s steps"],
     "notes": ["offset +3.75 s from 1186 dives", "per-dive IQR 2.25 to 5.25 s", "re-estimate after the shift: 0 s"]},
    {"kind": "proc", "title": ["Shift the ADCP and bottom-track times"], "fn": "shift_time",
     "detail": ["recompute CTD pressure and depth (gsw) at new times", "move bin_depths by the depth change"],
     "notes": ["depth change median +0.56 m", "1st–99th percentile −0.04 to +0.87 m"]},
    {"kind": "dec", "q": ["GDAC CTD gap", "> 60 s at the new time?"], "down": "no", "right": "yes",
     "side": {"title": ["Use OOINet's CTD pressure"], "detail": ["interpolated to the new time", "199 ensembles: dive starts at the surface"]}},
    {"kind": "group", "text": "Merge the science data onto the ADCP times"},
    {"kind": "proc", "title": ["Interpolate each GDAC variable"], "fn": "merge_datasets",
     "detail": ["split by sensor: ctd_, oxy_, flbbcd_, par_, glider_", "each from its own non-missing values"],
     "notes": ["use ctd_temperature/salinity/pressure:", "ADCP salinity is a constant 35,", "its temperature reads +2.7 °C warm"]},
    {"kind": "dec", "q": ["Gap in the variable", "> 60 s?"], "down": "no", "right": "yes",
     "side": {"title": ["Leave NaN"], "detail": ["no interpolation across the gap"]}},
    {"kind": "proc", "title": ["Correct w for the sound speed"], "fn": "sound_speed_factor · correct_vertical_velocity",
     "detail": ["phased array: only the DVL-axis component scales", "axis = glider attitude + 11° forward mount", "u, v unchanged; rebuilt from water_velocity_up"],
     "notes": ["DVL c − CTD c: median +8.3 m/s", "factor median 0.9927", "Δw cell 1: median −0.54 mm/s"]},
    {"kind": "proc", "title": ["Split the ADCP dives"], "fn": "add_profiles",
     "detail": ["new dive where |Δz| > 2 m and Δt > 4 s"], "notes": ["1388 ADCP profiles"]},
    {"kind": "proc", "title": ["Vendor QC flag per cell"], "fn": "qc.percent_good · correlation · error_velocity",
     "detail": ["percent good ≥ 63 good, < 50 fail", "correlation ≥ 115 good, < 63 fail", "error velocity ≤ 6.3 good, ≥ 12.6 cm/s fail"],
     "notes": ["merged → vendor_qc_flag (1, 3, 4, 9)"]},
    {"kind": "group", "text": "Engineering data and depth-averaged current"},
    {"kind": "dec", "q": ["Recovered engineering", "CSV exists?"], "down": "yes", "right": "no",
     "side": {"title": ["Use the telemetered CSV"]}},
    {"kind": "proc", "title": ["Add the waypoints"], "fn": "add_waypoints",
     "detail": ["each applies until the next is set", "(0, 0) placeholder → NaN"]},
    {"kind": "proc", "title": ["Depth-averaged current per surfacing"], "fn": "surfacing_currents · add_depth_averaged_current",
     "detail": ["(GPS fix − dead-reckoned position) / time", "+ current used in the dead reckoning", "given to every time from dive start to surfacing"],
     "notes": ["201 surfacings", "depends on m_speed"]},
    {"kind": "group", "text": "Glider dataset: dives, climbs, surfacings"},
    {"kind": "proc", "title": ["Build on the engineering time base"], "fn": "glider_dataset · add_flight_variables",
     "detail": ["~4 s, 646,483 times; GDAC interpolated as above", "pitch, heading, buoyancy change, m_speed, m_depth", "+ waypoints, depth-averaged current"]},
    {"kind": "proc", "title": ["List the science profiles"], "fn": "science_profiles",
     "detail": ["dive if the CTD pressure ends deeper than it starts"], "notes": ["1385 dives, 1385 climbs"]},
    {"kind": "dec", "q": ["Time inside a", "science profile?"], "down": "no", "right": "yes",
     "side": {"title": ["phase = dive or climb"], "fn": "add_phases", "detail": ["science_profile_id from the GDAC"]}},
    {"kind": "dec", "q": ["Gap between profiles", "≥ 60 s?"], "down": "no", "right": "yes",
     "side": {"title": ["phase = surface"], "detail": ["200 surfacings, median 22 s from", "the dead-reckoned surfacing time"]}},
    {"kind": "proc", "title": ["phase = unlabeled"], "detail": ["~1 s inflections; before the first or after", "the last profile (incl. the recovery surfacing)"]},
    {"kind": "proc", "title": ["Link ADCP to science profiles"], "fn": "link_profiles · add_profile_links",
     "detail": ["largest overlap in time", "ADCP gets science_profile_id, phase", "glider gets adcp_profile_id (whole dive)"],
     "notes": ["1385 dives: one ADCP profile each", "median overlap 99.4%", "ADCP 80, 401, 759 → climbs (6–10 samples)"]},
    {"kind": "data", "title": "Saved to data/processed", "files": ["merged.nc", "glider.nc", "depth_averaged_current.csv", "science_profiles.csv", "profile_links.csv"]},
]

# ---------------------------------------------------------------- notebook 4
NB4 = [
    {"kind": "data", "title": "Load", "files": ["merged.nc (glider), glider.nc (glider_ts)", "depth_averaged_current.csv", "adcp_bottom_track.nc, shifted +3.75 s", "eng: pitch, heading, buoyancy, m_speed"]},
    {"kind": "group", "text": "Flight model (gliderflight, steady state)"},
    {"kind": "proc", "title": ["Steady-flight mask (both datasets)"], "fn": "calibration_mask",
     "detail": ["depth ≥ 10 m and ≥ 5 m above the profile's deepest", "|pitch| ≥ 15°, |d buoyancy / dt| ≤ 0.5 cm³/s"],
     "notes": ["90% of DVL ensembles", "glider: 91% of dives, 70% of climbs"]},
    {"kind": "proc", "title": ["For each trial pitch offset (−6° to +4°)"], "fn": "calibrate_pitch_offset",
     "detail": ["fit Vg, Cd0 to the depth rate on dives + climbs", "solve on the DVL times", "model − DVL cell 2 horizontal speed on dives"],
     "notes": ["mass 65.480 kg, lift at default (ah 3.8)", "depth rate alone prefers ≈ +1°"]},
    {"kind": "dec", "q": ["Speed difference", "changes sign?"], "down": "yes", "right": "no",
     "side": {"title": ["Stop: ValueError"], "detail": ["widen the trial offsets"], "stop": True}},
    {"kind": "proc", "title": ["Interpolate the zero crossing"], "notes": ["pitch offset −2.00°"]},
    {"kind": "proc", "title": ["Calibrate weekly, solve on both datasets"], "fn": "calibrate_flight_model · solve_flight_model",
     "detail": ["whole deployment + each 7-day window", "windows < 1000 points use the deployment fit"],
     "notes": ["Vg 0.0639 m³, Cd0 0.134 (weekly 0.131–0.136)", "depth-rate bias: dives −0.03, climbs +0.93 cm/s", "dives-only fit for comparison: climbs +9.2 cm/s"]},
    {"kind": "proc", "title": ["Check 1: model vs DVL cell 2"], "fn": "dvl_glider_velocity · flight_model_skill",
     "detail": ["forward speed agrees by construction"],
     "notes": ["east/north r 0.97 / 0.93, rms ≈ 6 cm/s", "m_speed 0.308 vs model 0.284 m/s"]},
    {"kind": "group", "text": "Bottom tracking"},
    {"kind": "proc", "title": ["Add, correct, remove, bin"], "fn": "add_/remove_bottom_track · bin_profiles",
     "detail": ["nearest ensemble within 1 s, mm/s → m/s", "w sound-speed corrected like the water track", "DVL relative + BT → absolute cells, 2 m bins"],
     "notes": ["BT only near the bottom: 1292 of 1388 dives,", "the deepest ~64 m (median), from 53% of", "the dive's depth; 23% of all ensembles"]},
    {"kind": "proc", "title": ["Check 2: speed through water from BT"], "fn": "through_water_velocity",
     "detail": ["Method 2 with flight_weight = 0", "BT ground velocity − profile at the glider's depth", "(water measured earlier by deeper cells)"],
     "notes": ["99,225 ensembles: BT speed 0.256 m/s", "model +1.0, m_speed +2.2 cm/s", "cell 1 −1.2; cells 2–4 within 0.7 cm/s", "(lower part of the dives only)"]},
    {"kind": "group", "text": "Inversion, for each dive (LADCP-style least squares)"},
    {"kind": "proc", "title": ["Equations"], "fn": "invert_profiles",
     "detail": ["DVL: u(cell) − g(t) = relative velocity", "flight: g(t) − u(z_glider) = model (weight 1)", "smoothness 0.1 · unknowns u(2 m bins), g(t)"]},
    {"kind": "dec", "q": ["Which reference?"], "down": "Method 1: DAC", "right": "Method 2",
     "side": {"title": ["Bottom track only"], "detail": ["g(t) = −BT (weight 10), dac_weight 0", "only dives with bottom tracking"]}},
    {"kind": "proc", "title": ["Depth-averaged current"], "detail": ["mean u at the glider = DAC (weight 1000)"]},
    {"kind": "dec", "q": ["Cells and a reference", "in this dive?"], "down": "yes", "right": "no",
     "side": {"title": ["Skip the dive"], "stop": True}},
    {"kind": "dec", "q": ["Cells link every bin?", "(n_segments = 1)"], "down": "yes", "right": "no",
     "side": {"title": ["Skip the dive"], "detail": ["only the flight model would link the", "segments; its errors add up with depth"], "stop": True}},
    {"kind": "proc", "title": ["Solve (Schur complement on g)"], "detail": ["glider bins without cells → nearest observed bin"],
     "notes": ["Method 1: 1015 of 1388 dives", "Method 2: 981 of 1388 dives"]},
    {"kind": "group", "text": "Checks of the profiles"},
    {"kind": "proc", "title": ["Check 3: frame vs GPS"], "fn": "compare_depth_averaged_current",
     "detail": ["rotation from Method 2 to the DAC"], "notes": ["median −2.4°, IQR −10.4 to +5.2°", "(OOINet's rotation would give ≈ −14°)"]},
    {"kind": "proc", "title": ["Check 4: Method 2 vs DAC"], "detail": ["each surfacing, ≥ 90% coverage", "DAC also redone with the model speed", "over dives + climbs (glider dataset)"],
     "notes": ["74 surfacings: median 2.2 cm/s", "with the model speed: 2.1 cm/s", "mean east bias 2.8 cm/s (open)"]},
    {"kind": "proc", "title": ["Check 5: Method 1 vs bottom track"], "fn": "compare_profiles",
     "detail": ["offset (reference) + shape (rms)"], "notes": ["976 dives: offset median 5.6 cm/s", "shape rms 2.4 E / 2.3 N cm/s", "(only where BT profiles exist: near bottom)"]},
    {"kind": "proc", "title": ["Check 6: carrying the reference up"], "fn": "split_by_depth",
     "detail": ["Method 2 referenced to the deepest 20 m of BT", "tested on the BT above, by height above it"],
     "notes": ["968 dives, median error by height:", "0–10 m 3.2 · 10–20 m 4.6 · 20–30 m 5.4 cm/s", "30–40 m 6.1 · 40–50 m 6.9 cm/s", "mean east/north within ±0.6 cm/s"]},
    {"kind": "data", "title": "Saved to data/processed", "files": ["velocity_profiles_dac.nc", "velocity_profiles_bottom_track.nc", "bottom_track_profiles.nc", "flight_calibration.csv", "depth_averaged_current_check.csv"]},
]


# ---------------------------------------------------------------- page
CSS = """
:root {
  /* Layout: one reading column; each flowchart scrolls sideways in its own frame on phones */
  --bg: #f3f6f7; --fg: #13212a; --muted: #566872; --line: #b3c2ca; --panel: #ffffff;
  --accent: #0a6f7d; --dec: #e3f2f3; --data: #e8eef8; --datastroke: #4a6a9c;
  --stop: #b0491a; --stopfill: #fbeee7; --side: #f7fafb;
  --display: "Space Grotesk", "Segoe UI", system-ui, sans-serif;
  --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0d1418; --fg: #e2ebef; --muted: #93a6b0; --line: #3a4b54; --panel: #152027;
  --accent: #4fbfcc; --dec: #13303a; --data: #182536; --datastroke: #86a6d8;
  --stop: #ec8f5c; --stopfill: #2c1d16; --side: #111b20; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0d1418; --fg: #e2ebef; --muted: #93a6b0; --line: #3a4b54; --panel: #152027;
  --accent: #4fbfcc; --dec: #13303a; --data: #182536; --datastroke: #86a6d8;
  --stop: #ec8f5c; --stopfill: #2c1d16; --side: #111b20; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.55; }
main { max-width: 900px; margin: 0 auto; padding-block: 40px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1, h2 { font-family: var(--display); text-wrap: balance; letter-spacing: -0.01em; }
h1 { font-size: 2.1rem; margin: 0 0 6px; }
h2 { font-size: 1.35rem; margin: 48px 0 6px; }
.eyebrow { font: 600 12px/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); margin: 0 0 10px; }
p { max-width: 68ch; margin: 0 0 12px; }
.lede { color: var(--muted); font-size: 1.05rem; }
code { font-family: var(--mono); font-size: 0.88em; }
figure { margin: 18px 0 0; }
.frame { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); padding: 8px; }
figcaption { color: var(--muted); font-size: 13.5px; margin-top: 8px; max-width: 75ch; }
.chart { display: block; width: 100%; height: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 10px 22px; margin: 18px 0 0; padding: 0; list-style: none; font-size: 13.5px; color: var(--muted); }
.legend li { display: flex; align-items: center; gap: 8px; }
.sw { width: 26px; height: 16px; border-radius: 4px; border: 1.5px solid var(--line); background: var(--panel); flex: none; }
.sw.dec { transform: rotate(45deg) scale(0.72); background: var(--dec); border-color: var(--accent); border-radius: 2px; }
.sw.data { border-radius: 9px; background: var(--data); border-color: var(--datastroke); }
.sw.side { background: var(--side); border-style: dashed; }
.sw.stop { background: var(--stopfill); border-color: var(--stop); }
.sw.next { border-style: dashed; border-color: var(--accent); background: transparent; }
.issues { display: grid; gap: 10px; margin: 14px 0 0; padding: 0; list-style: none; }
.issues li { background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--stop); border-radius: 6px; padding: 10px 14px; max-width: 80ch; }
.issues b { font-family: var(--display); font-weight: 600; }
/* svg */
.n-proc { fill: var(--panel); stroke: var(--line); stroke-width: 1.5; }
.n-nb { fill: var(--panel); stroke: var(--accent); stroke-width: 2; }
.n-nb3 { fill: var(--panel); stroke: var(--line); stroke-width: 1.5; stroke-dasharray: 5 4; }
.n-next { fill: none; stroke: var(--accent); stroke-width: 1.5; stroke-dasharray: 6 5; }
.n-dec { fill: var(--dec); stroke: var(--accent); stroke-width: 1.5; }
.n-data { fill: var(--data); stroke: var(--datastroke); stroke-width: 1.3; }
.n-side { fill: var(--side); stroke: var(--line); stroke-width: 1.3; stroke-dasharray: 4 3; }
.n-stop { fill: var(--stopfill); stroke: var(--stop); stroke-width: 1.4; }
.rule { stroke: var(--line); stroke-width: 1; stroke-dasharray: 2 4; }
.e, .e-soft, .e-dash, .e-route { stroke: var(--muted); stroke-width: 1.4; fill: none; }
.e-soft { stroke-width: 1; opacity: 0.75; }
.e-dash { stroke-dasharray: 5 4; }
.e-route { stroke: var(--datastroke); stroke-dasharray: 3 3; }
.ahd { fill: var(--muted); }
text { fill: var(--fg); font-family: var(--body); }
.tt { font-size: 13.5px; font-weight: 600; }
.nbt { font-family: var(--display); font-size: 15px; font-weight: 600; }
.fn { font-family: var(--mono); font-size: 11.5px; fill: var(--accent); }
.fl { font-family: var(--mono); font-size: 12px; }
.dt { font-size: 12px; fill: var(--muted); }
.dq { font-size: 13px; font-weight: 600; }
.el { font-family: var(--mono); font-size: 11.5px; fill: var(--accent); }
.nt { font-size: 12px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.gh { font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.08em; fill: var(--accent); font-weight: 600; }
"""

LEGEND = """<ul class="legend" aria-label="Legend">
<li><span class="sw"></span>processing step, with its function</li>
<li><span class="sw dec"></span>decision</li>
<li><span class="sw side"></span>branch taken</li>
<li><span class="sw stop"></span>stop or skip</li>
<li><span class="sw data"></span>files read or written</li>
<li><span class="sw next"></span>not built yet</li>
</ul>"""


def figure(svg_text, caption):
    return f'<figure><div class="frame">{svg_text}</div><figcaption>{escape(caption)}</figcaption></figure>'


def page():
    return f"""<title>GL379 DVL Pipeline</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>{CSS}</style>
<main>
<p class="eyebrow">ooicgsn_glider_dvl · CP05MOAS-GL379 deployment 8</p>
<h1>DVL Processing Pipeline</h1>
<p class="lede">How the OOI Pioneer glider DVL data goes from OOINet downloads to absolute velocity profiles, as the code stands on 20 November 2026. Numbers beside the steps are from GL379 deployment 8 (27 Sep to 18 Nov 2019).</p>
{LEGEND}

<h2>Data flow between the notebooks</h2>
<p>Notebook 2 builds two datasets from the same downloads: one on the DVL times, which only cover the dives, and one on the glider's engineering times, which cover the dives, climbs, and surfacings. Notebook 4 uses both, and reads the raw bottom track and engineering data again.</p>
{figure(overview(), "Each notebook reads the files above it. Notebooks 3 and 3.1 are exploratory and don't feed notebook 4, which also reads the raw bottom track and engineering data directly.")}

<h2>1 · Data access and download</h2>
<p>Runs on the OOI JupyterHub, where <code>load_kdata</code> works. Each deployment on the FZ line is downloaded in three sections, each logged as ok, partial, missing, or failed.</p>
{figure(flowchart(NB1, "Notebook 1 download flow with its decisions"), "The engineering data falls back to the telemetered stream when the recovered stream is missing; a failed GDAC download is logged and retried on the next run.")}

<h2>2 · Process and merge</h2>
<p>Corrects the OOINet ADCP data, merges the glider CTD onto the DVL times, and builds the glider dataset with its dives, climbs, and surfacings, linked to the ADCP dives.</p>
{figure(flowchart(NB2, "Notebook 2 processing flow with its decisions"), "The order matters: the clock offset is corrected before the merge, so the CTD is interpolated at the corrected DVL times, and the sound-speed correction needs the merged CTD.")}

<h2>4 · Remove the glider motion</h2>
<p>Calibrates the flight model on dives and climbs with a pitch offset, then solves each dive for the absolute velocity profile two ways, referenced to the depth-averaged current (Method 1) or to bottom tracking alone (Method 2), and checks each against the other.</p>
{figure(flowchart(NB4, "Notebook 4 flight model, inversion, and checks with their decisions"), "Checks 1 and 2 test the flight model; checks 3 to 6 test the profiles. Bottom tracking only covers the deepest part of each dive, so checks 2, 5, and 6 describe the lower water column; only check 4 covers the whole dive, and only on average.")}

<h2>Open questions</h2>
<ul class="issues">
<li><b>Eastward bias against the depth-averaged current.</b> Method 2's mean eastward difference from the DAC is 2.8 cm/s (0.8 before the pitch offset), though the median magnitude improved to 2.2 cm/s. Possibly a few large differences in the strong currents at the end of the deployment.</li>
<li><b>Stale attitude in the DVL rotation.</b> The DVL rotated its velocities with pitch and heading about 9 s old. Not corrected; it matters during turns and pitch changes.</li>
<li><b>Pitch offset vs depth rate.</b> The −2° offset comes from the horizontal speed; the depth rate alone prefers about +1° and its rms grows by 0.1 cm/s at −2°.</li>
<li><b>Vertical sound-speed correction.</b> Assumes the DVL firmware uses the beam angle for its own sound speed; the other assumption changes Δw by about a quarter (0.1–0.2 mm/s).</li>
<li><b>The upper half of the Method 2 profiles.</b> Above the bottom tracking, no check tests a single dive: the error grows from 3.2 cm/s just above the reference to 6.9 cm/s 40–50 m above it, and most dives extend another 60–70 m above their bottom tracking. The mean errors stay within ±0.6 cm/s, so averages over many dives should hold up better than single profiles.</li>
<li><b>Dives that aren't solved.</b> 373 dives (Method 1) have cells that don't link every depth bin, mostly deep dives with few scatterers.</li>
</ul>
</main>
"""

if __name__ == "__main__":
    out = HERE / "dvl-pipeline.html"
    out.write_text(page(), encoding="utf8")
    print(f"wrote {out}")
