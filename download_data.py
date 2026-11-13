"""Download the BEELINE mESC scRNA-seq data and ground truth from Zenodo (record 3701939, v3).

Usage:  python download_data.py [output_dir]      (default: ./data)
Extracts: ExpressionData.csv, GeneOrdering.csv, PseudoTime.csv (mESC),
          mESC-ChIP-seq-network.csv, mouse-tfs.csv
"""
import hashlib, os, shutil, sys, urllib.request, zipfile

ZENODO = "https://zenodo.org/records/3701939/files/{}?download=1"
ARCHIVES = {   # name: published md5
    "BEELINE-data.zip":     "52581e7e27aa699310384238daac4751",
    "BEELINE-Networks.zip": "a7dca84d302bb127bad962ae52815662",
}
WANTED = {     # file name inside the archive -> must appear in this path fragment
    "ExpressionData.csv":        "scRNA-Seq/mESC/",
    "GeneOrdering.csv":          "scRNA-Seq/mESC/",
    "PseudoTime.csv":            "scRNA-Seq/mESC/",
    "mESC-ChIP-seq-network.csv": "Networks/mouse/",
    "mouse-tfs.csv":             "",
}
OPTIONAL = {   # extracted if present (second ground truth); skipped otherwise
    "STRING-network.csv":        "mouse",
}

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main(out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    local = os.environ.get("BEELINE_LOCAL_DIR")      # optional: use already-downloaded zips
    for name, checksum in ARCHIVES.items():
        zpath = os.path.join(out_dir, name)
        if not os.path.exists(zpath):
            if local:
                shutil.copy(os.path.join(local, name), zpath)
            else:
                print(f"downloading {name} ...", flush=True)
                urllib.request.urlretrieve(ZENODO.format(name), zpath)
            if not local and md5(zpath) != checksum:
                os.remove(zpath)
                raise RuntimeError(f"MD5 mismatch for {name}; download again")
        with zipfile.ZipFile(zpath) as z:
            for member in z.namelist():
                base = os.path.basename(member)
                frag = WANTED.get(base, OPTIONAL.get(base))
                if frag is not None and frag in member and "__MACOSX" not in member:
                    target = os.path.join(out_dir, base)
                    if not os.path.exists(target):
                        with z.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    print("extracted", member)
    missing = [f for f in WANTED if not os.path.exists(os.path.join(out_dir, f))]
    if missing:
        raise RuntimeError(f"not found in the archives: {missing}")
    for f in OPTIONAL:
        print(f"optional {f}:", "found" if os.path.exists(os.path.join(out_dir, f)) else "not found (skipped)")
    print("all required files present in", out_dir)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
