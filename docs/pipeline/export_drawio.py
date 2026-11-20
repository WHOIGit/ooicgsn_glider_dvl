"""Export the pipeline flowcharts in build.py to a draw.io file, one page per diagram.

The draw.io file is a starting point for editing by hand: once it has been edited, it is the
source of those diagrams, and running this again overwrites the edits (it refuses unless
--force is given).

    python export_drawio.py [--force]
"""

from html import escape
import sys
import xml.etree.ElementTree as ET

from build import CX, GAP, HERE, NB1, NB2, NB4, RIGHT_W, RIGHT_X, SPINE_W, SPINE_X, W, proc_height

# Light-theme colors from the HTML page
FG, MUTED, LINE, ACCENT = "#13212a", "#566872", "#b3c2ca", "#0a6f7d"
MONO = "Courier New"

STYLES = {
    "proc": f"rounded=1;arcSize=4;fillColor=#ffffff;strokeColor={LINE};strokeWidth=1.5;",
    "side": f"rounded=1;arcSize=4;fillColor=#f7fafb;strokeColor={LINE};dashed=1;",
    "stop": "rounded=1;arcSize=4;fillColor=#fbeee7;strokeColor=#b0491a;strokeWidth=1.4;",
    "data": "rounded=1;arcSize=20;fillColor=#e8eef8;strokeColor=#4a6a9c;",
    "dec": f"rhombus;fillColor=#e3f2f3;strokeColor={ACCENT};strokeWidth=1.5;",
    "nb": f"rounded=1;arcSize=6;fillColor=#ffffff;strokeColor={ACCENT};strokeWidth=2;",
    "nb3": f"rounded=1;arcSize=6;fillColor=#ffffff;strokeColor={LINE};dashed=1;",
    "next": f"rounded=1;arcSize=6;fillColor=none;strokeColor={ACCENT};dashed=1;",
}
TEXT = f"html=1;whiteSpace=wrap;fontColor={FG};fontSize=12;"
BOX = TEXT + "align=left;verticalAlign=top;spacingLeft=8;spacingTop=2;"
EDGE = (
    f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;endSize=5;"
    f"strokeColor={MUTED};fontColor={ACCENT};fontFamily={MONO};fontSize=11;"
    "labelBackgroundColor=#ffffff;"
)


def label(title=(), fn=None, detail=(), files=()):
    """An HTML label: bold title lines, the function in monospace, muted detail lines."""
    lines = [f"<b>{escape(s)}</b>" for s in title]
    if fn:
        lines.append(f'<font face="{MONO}" color="{ACCENT}" style="font-size:11px">{escape(fn)}</font>')
    lines += [f'<font face="{MONO}" style="font-size:11px">{escape(s)}</font>' for s in files]
    lines += [f'<font color="{MUTED}" style="font-size:11px">{escape(s)}</font>' for s in detail]
    return "<br>".join(lines)


class Page:
    def __init__(self, name):
        self.name, self.cells, self.n = name, [], 0

    def _id(self):
        self.n += 1
        return f"{self.name[:3].lower()}{self.n}"

    def vertex(self, x, y, w, h, value, style):
        cid = self._id()
        self.cells.append(("v", cid, value, style, (x, y, w, h)))
        return cid

    def edge(self, source, target, value="", style="", exit_=(0.5, 1), entry=(0.5, 0)):
        cid = self._id()
        pts = f"exitX={exit_[0]};exitY={exit_[1]};entryX={entry[0]};entryY={entry[1]};"
        self.cells.append(("e", cid, value, EDGE + pts + style, (source, target)))
        return cid

    def text(self, x, y, w, h, lines, style=""):
        value = "<br>".join(escape(s) for s in lines)
        return self.vertex(x, y, w, h, value, TEXT + "text;align=left;verticalAlign=top;"
                           f"fontColor={MUTED};fontSize=11;" + style)

    def heading(self, x, y, text):
        return self.vertex(x, y, 400, 20, escape(text.upper()), TEXT + "text;align=left;"
                           f"fontFamily={MONO};fontSize=11;fontStyle=1;fontColor={ACCENT};")

    def xml(self, diagram):
        model = ET.SubElement(diagram, "mxGraphModel", grid="1", gridSize="10", guides="1",
                              tooltips="1", connect="1", arrows="1", fold="1", page="0",
                              pageScale="1", math="0", shadow="0")
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")
        for kind, cid, value, style, geo in self.cells:
            if kind == "v":
                cell = ET.SubElement(root, "mxCell", id=cid, value=value, style=style,
                                     vertex="1", parent="1")
                x, y, w, h = geo
                ET.SubElement(cell, "mxGeometry", x=f"{x:.0f}", y=f"{y:.0f}", width=f"{w:.0f}",
                              height=f"{h:.0f}", **{"as": "geometry"})
            else:
                cell = ET.SubElement(root, "mxCell", id=cid, value=value, style=style, edge="1",
                                     parent="1", source=geo[0], target=geo[1])
                ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})


def flowchart(page, nodes, title, caption):
    """The same layout as build.flowchart, as draw.io cells with connected edges."""
    page.vertex(SPINE_X, -40, W - 2 * SPINE_X, 28, f"<b>{escape(title)}</b>",
                TEXT + "text;align=left;fontSize=18;")
    y, prev, down = 16, None, None
    for n in nodes:
        kind = n["kind"]
        if kind == "group":
            y += 8
            page.vertex(20, y - 1, W - 40, 2, "", f"line;strokeColor={LINE};dashed=1;dashPattern=2 4;")
            page.heading(CX + 14, y + 6, n["text"])
            y += 34
            continue
        side_h = 0
        if kind in ("proc", "data"):
            if kind == "proc":
                h = proc_height(n)
                cid = page.vertex(SPINE_X, y, SPINE_W, h, label(n.get("title", []), n.get("fn"),
                                  n.get("detail", [])), BOX + STYLES["proc"])
            else:
                h = 16 + 16 * len(n["files"]) + (17 if n.get("title") else 0)
                title = [n["title"]] if n.get("title") else []
                cid = page.vertex(SPINE_X, y, SPINE_W, h, label(title, files=n["files"]),
                                  BOX + STYLES["data"] + "spacingLeft=10;")
            if n.get("notes"):
                side_h = 15 * len(n["notes"]) + 10
                page.text(RIGHT_X, y, RIGHT_W, side_h, n["notes"])
            next_y = y + max(h, side_h) + GAP
        else:  # decision
            h = 74
            mid = y + h / 2
            q = "<br>".join(escape(s) for s in n["q"])
            cid = page.vertex(SPINE_X, y, SPINE_W, h, f"<b>{q}</b>",
                              TEXT + "align=center;verticalAlign=middle;" + STYLES["dec"])
            side = n["side"]
            sh = proc_height(side)
            sid = page.vertex(RIGHT_X, mid - 20, RIGHT_W, sh,
                              label(side.get("title", []), side.get("fn"), side.get("detail", [])),
                              BOX + STYLES["stop" if side.get("stop") else "side"])
            page.edge(cid, sid, escape(n["right"]), exit_=(1, 0.5), entry=(0, 20 / sh))
            next_y = max(y + h, mid - 20 + sh) + GAP
        if prev is not None:
            page.edge(prev, cid, escape(down or ""))
        down = n.get("down")
        prev, y = cid, next_y
    page.text(SPINE_X, y, W - 2 * SPINE_X, 60, [caption], f"fontColor={MUTED};fontSize=12;fontStyle=2;")


def overview(page):
    """The same layout as build.overview."""
    cw, ch = 178, 62
    page.vertex(20, -40, 900, 28, "<b>Data flow between the notebooks</b>",
                TEXT + "text;align=left;fontSize=18;")

    def nb(x, y, w, title, subs, style="nb"):
        h = 34 + 17 * len(subs)
        value = f'<b style="font-size:14px">{escape(title)}</b><br>' + "<br>".join(
            f'<font color="{MUTED}" style="font-size:11px">{escape(s)}</font>' for s in subs)
        return page.vertex(x, y, w, h, value, TEXT + STYLES[style]), y + h

    def chip(x, y, name, desc):
        value = (f'<font face="{MONO}">{"<br>".join(escape(s) for s in name.split("|"))}</font><br>'
                 f'<font color="{MUTED}" style="font-size:11px">{escape(desc)}</font>')
        return page.vertex(x, y, cw, ch, value, TEXT + STYLES["data"])

    soft = f"strokeColor={LINE};"
    xs = [20 + i * (cw + 12) for i in range(5)]
    nb1, nb1_bottom = nb(330, 16, 320, "1 · Data access and download",
                         ["OOINet M2M · THREDDS · IOOS GDAC ERDDAP"])
    raw_y = nb1_bottom + 56
    page.heading(20, raw_y - 26, "data/raw")
    raw = [chip(x, raw_y, a, b) for x, (a, b) in zip(xs, [
        ("adcp_data.nc", "DVL water track"),
        ("adcp_bottom_track.nc", "DVL bottom track"),
        ("science.nc", "GDAC CTD, profiles"),
        ("eng_recovered.csv", "glider engineering"),
        ("mooring_locations.csv", "Pioneer moorings"),
    ])]
    for c in raw:
        page.edge(nb1, c, style=soft)

    nb2, nb2_bottom = nb(330, raw_y + ch + 56, 320, "2 · Process and merge",
                         ["corrections, merge, profiles, links"])
    for c in raw[:4]:
        page.edge(c, nb2, style=soft)

    pro_y = nb2_bottom + 56
    page.heading(20, pro_y - 26, "data/processed")
    pro = [chip(x, pro_y, a, b) for x, (a, b) in zip(xs, [
        ("merged.nc", "ADCP times · dives"),
        ("glider.nc", "eng times · dives+climbs"),
        ("depth_averaged_|current.csv", "201 surfacings"),
        ("science_profiles.csv", "1385 dives, 1385 climbs"),
        ("profile_links.csv", "ADCP ↔ science"),
    ])]
    for c in pro:
        page.edge(nb2, c, style=soft)

    nb4_y = pro_y + ch + 56
    nb3, _ = nb(20, nb4_y, 300, "3 · 3.1 · Data analysis",
                ["exploratory: track maps, QC, simple bins"], "nb3")
    nb4, nb4_bottom = nb(400, nb4_y, 420, "4 · Remove the glider motion",
                         ["flight model, bottom track, inversion, checks",
                          "also reads the raw bottom track and eng data"])
    page.edge(pro[0], nb3, style=soft)
    for c in pro[:3]:
        page.edge(c, nb4, style=soft)

    out_y = nb4_bottom + 56
    page.heading(20, out_y - 26, "data/processed")
    outs = [chip(x, out_y, a, b) for x, (a, b) in zip(xs, [
        ("velocity_|profiles_dac.nc", "Method 1"),
        ("velocity_profiles_|bottom_track.nc", "Method 2"),
        ("bottom_track_|profiles.nc", "BT-referenced, binned"),
        ("flight_calibration.csv", "Vg, Cd0, pitch offset"),
        ("depth_averaged_|current_check.csv", "per surfacing"),
    ])]
    for c in outs:
        page.edge(nb4, c, style=soft)

    nx_y = out_y + ch + 50
    nxt = page.vertex(330, nx_y, 320, 50, "<b>Next · objective mapping of sections</b><br>"
                      f'<font color="{MUTED}" style="font-size:11px">not built yet</font>',
                      TEXT + STYLES["next"])
    for c in outs[:2]:
        page.edge(c, nxt, style="dashed=1;")

    # Legend, to the right
    lx, ly = 1010, 16
    page.heading(lx, ly, "legend")
    for i, (style, text) in enumerate([
        ("proc", "processing step, with its function"),
        ("dec", "decision"),
        ("side", "branch taken"),
        ("stop", "stop or skip"),
        ("data", "files read or written"),
        ("next", "not built yet"),
    ]):
        y = ly + 30 + 34 * i
        page.vertex(lx, y, 36, 22, "", STYLES[style])
        page.text(lx + 46, y + 1, 220, 22, [text], f"fontColor={FG};fontSize=12;")


def main():
    out = HERE / "dvl-pipeline.drawio"
    if out.exists() and "--force" not in sys.argv:
        sys.exit(f"{out} exists and may have hand edits; use --force to overwrite it")

    pages = []
    p = Page("Overview")
    overview(p)
    pages.append(p)
    for name, nodes, title, caption in [
        ("Notebook 1", NB1, "1 · Data access and download",
         "The engineering data falls back to the telemetered stream when the recovered stream "
         "is missing; a failed GDAC download is logged and retried on the next run."),
        ("Notebook 2", NB2, "2 · Process and merge",
         "The order matters: the clock offset is corrected before the merge, so the CTD is "
         "interpolated at the corrected DVL times, and the sound-speed correction needs the "
         "merged CTD."),
        ("Notebook 4", NB4, "4 · Remove the glider motion",
         "Checks 1 and 2 test the flight model; checks 3 to 6 test the profiles. Bottom tracking "
         "only covers the deepest part of each dive, so checks 2, 5, and 6 describe the lower "
         "water column; only check 4 covers the whole dive, and only on average."),
    ]:
        p = Page(name)
        flowchart(p, nodes, title, caption)
        pages.append(p)

    mxfile = ET.Element("mxfile", host="Electron", type="device")
    for i, p in enumerate(pages):
        p.xml(ET.SubElement(mxfile, "diagram", id=f"page{i}", name=p.name))
    ET.indent(mxfile)
    ET.ElementTree(mxfile).write(out, encoding="utf-8", xml_declaration=False)
    print(f"wrote {out} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
