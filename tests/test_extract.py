"""Tests for scripts/unzip_archive.py.

The extractor writes into the user's download folder, so its failure modes are
data loss and a full disk, not just a wrong number. Covered here: CRC-checked
extraction, re-run skipping and re-download detection, corrupt zips, path
traversal, in-progress downloads, the free-space floor, and the HEL1OS shared
date-tree layout -- where treating the first path component ("2024") as the
product made every new extraction delete all earlier ones of that year.

    python -m tests.test_extract
"""

import os
import pathlib
import subprocess
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
fails = []
def check(n, c, d=""):
    print(("  PASS  " if c else "  FAIL  ") + n + ("" if c else f"  {d}"))
    if not c:
        fails.append(n)

def mkzip(p, members):
    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)

def run(root, *extra):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "unzip_archive.py"), "--data-root", str(root), *extra],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr

with tempfile.TemporaryDirectory() as d:
    root = pathlib.Path(d)
    good = {f"AL1_SLX_L1_2026010{i}_v1.0/SDD2/AL1_SOLEXS_2026010{i}_SDD2_L1.pi.gz": os.urandom(200_000) for i in (1,)}
    mkzip(root / "dl/a/AL1_SLX_L1_20260101_v1.0.zip",
          {**good, "AL1_SLX_L1_20260101_v1.0/SDD1/AL1_SOLEXS_20260101_SDD1_L1.gti.gz": b"g"})
    mkzip(root / "dl/b/AL1_SLX_L1_20260102_v1.0.zip", {"AL1_SLX_L1_20260102_v1.0/SDD2/x.lc.gz": b"ok" * 1000})
    # corrupt: flip bytes inside the compressed data
    c = root / "dl/c/AL1_SLX_L1_20260103_v1.0.zip"
    mkzip(c, {"AL1_SLX_L1_20260103_v1.0/SDD2/y.pi.gz": os.urandom(300_000)})
    raw = bytearray(c.read_bytes())
    raw[200:260] = b"\x00" * 60
    c.write_bytes(raw)
    # path traversal
    mkzip(root / "dl/e/AL1_SLX_L1_20260104_v1.0.zip", {"../../evil.txt": b"pwned"})
    # in-progress download
    mkzip(root / "dl/f/tmp.zip", {"AL1_SLX_L1_20260105_v1.0/SDD2/z.pi.gz": b"z"})
    (root / "dl/f/tmp.zip").rename(root / "dl/f/AL1_SLX_L1_20260105_v1.0.zip.part")

    rc, out = run(root / "dl", "--dest", str(root / "out"))
    dest = root / "out"
    check("good zips extracted with their internal folder", (dest / "AL1_SLX_L1_20260101_v1.0/SDD2").is_dir()
          and (dest / "AL1_SLX_L1_20260102_v1.0/SDD2/x.lc.gz").exists(), out)
    ok_bytes = (dest / "AL1_SLX_L1_20260101_v1.0/SDD2/AL1_SOLEXS_20260101_SDD2_L1.pi.gz").read_bytes()
    check("extracted bytes identical to the zip member", ok_bytes == list(good.values())[0])
    check("markers written", (dest / "AL1_SLX_L1_20260101_v1.0/.extracted.json").exists())
    check("corrupt zip reported as failed, not extracted", "20260103" in out and "failed" in out
          and not (dest / "AL1_SLX_L1_20260103_v1.0").exists(), out)
    check("path traversal rejected and nothing written outside dest",
          not (root / "evil.txt").exists() and not (root.parent / "evil.txt").exists() and "unsafe path" in out, out)
    check(".zip.part ignored", "20260105" not in out and not (dest / "AL1_SLX_L1_20260105_v1.0").exists())
    check("no temporary folders left behind", not any(p.name.startswith(".tmp_") for p in dest.iterdir()))
    check("non-zero exit when something failed", rc == 1)

    rc2, out2 = run(root / "dl", "--dest", str(root / "out"))
    check("second run skips completed zips", "'skipped': 2" in out2, out2)

    z = root / "dl/a/AL1_SLX_L1_20260101_v1.0.zip"
    st = z.stat()
    os.utime(z, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    rc3, out3 = run(root / "dl", "--dest", str(root / "out"))
    check("a zip re-downloaded with identical content is NOT re-extracted",
          "'skipped': 2" in out3, out3)

    new_bytes = os.urandom(200_000)
    mkzip(z, {"AL1_SLX_L1_20260101_v1.0/SDD2/AL1_SOLEXS_20260101_SDD2_L1.pi.gz": new_bytes,
              "AL1_SLX_L1_20260101_v1.0/SDD1/AL1_SOLEXS_20260101_SDD1_L1.gti.gz": b"g"})
    rc3, out3 = run(root / "dl", "--dest", str(root / "out"))
    check("a zip whose content changed IS re-extracted, with the new bytes",
          "'ok': 1" in out3 and "'skipped': 1" in out3
          and (root / "out/AL1_SLX_L1_20260101_v1.0/SDD2/AL1_SOLEXS_20260101_SDD2_L1.pi.gz").read_bytes() == new_bytes,
          out3)

    rc4, out4 = run(root / "dl", "--dest", str(root / "out2"), "--min-free-gb", "100000")
    check("disk guard stops extraction instead of filling the disk", "no-space" in out4
          and not (root / "out2" / "AL1_SLX_L1_20260101_v1.0").exists(), out4)
# --- HEL1OS layout: every zip unpacks into a shared YYYY/MM/DD tree --------------
with tempfile.TemporaryDirectory() as d:
    root = pathlib.Path(d)
    def hls(name, day, n=3):
        base = f"2024/06/{day}/{name}"
        return {f"{base}/czt/lightcurve_czt1.fits": os.urandom(50_000),
                f"{base}/aux/gticzt1.fits": b"gti",
                f"{base}/events/evt.fits": os.urandom(120_000)}
    mkzip(root / "dl/HLS_20240608_012518_38080sec_lev1_V111.zip", hls("HLS_20240608_012518_38080sec_lev1_V111", "08"))
    mkzip(root / "dl/HLS_20240608_130000_40000sec_lev1_V111.zip", hls("HLS_20240608_130000_40000sec_lev1_V111", "08"))
    mkzip(root / "dl/HLS_20240609_000000_43000sec_lev1_V111.zip", hls("HLS_20240609_000000_43000sec_lev1_V111", "09"))
    out = root / "out"
    rc, log = run(root / "dl", "--dest", str(out), "--instrument", "hel1os", "--members", "all")
    prods = sorted(p.name for p in out.rglob("HLS_*") if p.is_dir())
    check("HEL1OS: all three products survive in the shared date tree", len(prods) == 3, f"{prods} {log}")
    check("HEL1OS: event lists extracted when --members all",
          len(list(out.rglob("evt.fits"))) == 3)
    # Change one product's content so it genuinely has to be re-extracted.
    mkzip(root / "dl/HLS_20240608_012518_38080sec_lev1_V111.zip",
          hls("HLS_20240608_012518_38080sec_lev1_V111", "08"))
    rc, log = run(root / "dl", "--dest", str(out), "--instrument", "hel1os", "--members", "all")
    prods2 = sorted(p.name for p in out.rglob("HLS_*") if p.is_dir())
    check("HEL1OS: re-extracting one product does not delete its neighbours",
          prods2 == prods and "'ok': 1" in log and "'skipped': 2" in log, f"{prods2} {log}")
    rc, log = run(root / "dl", "--dest", str(root / "lc"), "--instrument", "hel1os")
    check("HEL1OS default (lightcurves) skips event lists",
          not list((root / "lc").rglob("evt.fits")) and len(list((root / "lc").rglob("lightcurve_czt1.fits"))) == 3, log)
    rc, log = run(root / "dl", "--dest", str(out), "--instrument", "hel1os", "--members", "lightcurves")
    check("HEL1OS: a light-curves pass never deletes a full extraction's event lists",
          len(list(out.rglob("evt.fits"))) == 3 and "'skipped': 3" in log, log)

# --- HEL1OS zip holding TWO products, next to another zip for the same day ----
with tempfile.TemporaryDirectory() as d:
    root = pathlib.Path(d)

    def prod(day, name):
        base = f"2026/07/{day}/{name}"
        return {f"{base}/czt/lightcurve_czt1.fits": os.urandom(40_000),
                f"{base}/aux/gticzt1.fits": b"gti"}
    multi = {**prod("05", "HLS_20260705_000010_6690sec_lev1_V111"),
             **prod("05", "HLS_20260705_030536_32057sec_lev1_V111")}
    mkzip(root / "dl/HLS_20260705_030536_32057sec_lev1_V111.zip", multi)
    mkzip(root / "dl/HLS_20260705_120000_40000sec_lev1_V111.zip",
          prod("05", "HLS_20260705_120000_40000sec_lev1_V111"))
    out = root / "out"
    rc, log = run(root / "dl", "--dest", str(out), "--instrument", "hel1os", "--members", "all")
    day = out / "2026/07/05"
    names = sorted(p.name for p in day.iterdir() if p.is_dir())
    check("multi-product zip: each product lands in its own folder beside the neighbour",
          len(names) == 3, f"{names} {log}")
    check("multi-product zip: every product folder gets a completion marker",
          all((day / n / ".extracted.json").exists() for n in names))
    check("no marker is written on the shared day folder", not (day / ".extracted.json").exists())
    # New content for the multi-product zip, so it must be re-extracted.
    mkzip(root / "dl/HLS_20260705_030536_32057sec_lev1_V111.zip",
          {**prod("05", "HLS_20260705_000010_6690sec_lev1_V111"),
           **prod("05", "HLS_20260705_030536_32057sec_lev1_V111")})
    rc, log = run(root / "dl", "--dest", str(out), "--instrument", "hel1os", "--members", "all")
    check("re-extracting the multi-product zip leaves the neighbour's data intact",
          (day / "HLS_20260705_120000_40000sec_lev1_V111/czt/lightcurve_czt1.fits").exists()
          and "'ok': 1" in log and "'skipped': 1" in log, log)

# --- the same product inside two different zips (as PRADAN really ships) ------
with tempfile.TemporaryDirectory() as d:
    root = pathlib.Path(d)
    shared = {"2026/07/05/HLS_20260705_000010_6690sec_lev1_V111/czt/lightcurve_czt1.fits": os.urandom(30_000),
              "2026/07/05/HLS_20260705_000010_6690sec_lev1_V111/aux/gticzt1.fits": b"gti"}
    other = {"2026/07/05/HLS_20260705_030536_32057sec_lev1_V111/czt/lightcurve_czt1.fits": os.urandom(30_000)}
    mkzip(root / "dl/HLS_20260705_000010_6690sec_lev1_V111.zip", shared)
    mkzip(root / "dl/HLS_20260705_030536_32057sec_lev1_V111.zip", {**shared, **other})
    out = root / "out"
    run(root / "dl", "--dest", str(out), "--instrument", "hel1os", "--members", "all", "--threads", "1")
    rc, log = run(root / "dl", "--dest", str(out), "--instrument", "hel1os", "--members", "all")
    check("a product shipped in two zips is not re-extracted on every run",
          "'skipped': 2" in log, log)

print("All extraction checks passed." if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
