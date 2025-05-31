import argparse
import hashlib
import io
import pathlib
import requests
import tqdm
import pandas as pd
from PIL import Image


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def download(url: str) -> bytes:
    return requests.get(url, timeout=20).content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with url,year columns")
    ap.add_argument("--out", required=True, help="Directory to store images")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load previous clean.csv if it exists to persist deduplication
    clean_csv_path = pathlib.Path("data/clean.csv")
    seen = set()
    rows = []

    if clean_csv_path.exists():
        prev_df = pd.read_csv(clean_csv_path)
        seen.update(pathlib.Path(p).stem for p in prev_df["path"])
        rows.extend(prev_df.to_dict("records"))

    df = pd.read_csv(args.csv)

    iters_to_write = 100
    i = len(rows)

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        try:
            data = download(row["URL"])
            h = sha1(data)
            if h in seen:
                continue
            seen.add(h)

            fn = out / f"{h}.jpg"
            with Image.open(io.BytesIO(data)) as im:
                im.convert("RGB").save(fn, "JPEG", quality=95)
            rows.append({"path": str(fn), "year": int(row["Year"])})

            i += 1
            if i % iters_to_write == 0:
                pd.DataFrame(rows).to_csv(clean_csv_path, index=False)

        except Exception as e:
            tqdm.tqdm.write(f"skip {row['URL']} : {e}")

    pd.DataFrame(rows).to_csv(clean_csv_path, index=False)


if __name__ == "__main__":
    main()
