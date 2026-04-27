import time
import cv2
from fast_alpr import ALPR
import numpy as np
import pandas as pd
from pandas.io.formats import style
import onnxruntime as ort
from my_logger import \
setup_loggers, teardown_loggers, export_html, mismatch_table, verify_ocr
from vlogging import VisualRecord # modify __init__ in git
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
import re


def analyze(img_path, alpr, loggers, og_lookup):
    name = img_path.name
    frame = cv2.imread(str(img_path))
    if frame is None:
        loggers["results"].info(f"{name}~READ_FAILURE")
        return

    og = og_lookup.get(name, {})
    og_ocr = str(og.get('LICENSEPLATE', 'NOT_FOUND')).strip() or "NULL"
    og_status = str(og.get('STATUS', 'NOT_FOUND')).strip() or "NULL"


# make image processing a new function
    og_h, og_w = frame.shape[:2]
    new_w = 3200
    new_h = int(og_h * (new_w / og_w))
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # doesn't seem to be helping much...
    #blur = cv2.GaussianBlur(frame, (0, 0), 3)
    #sharp = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)
    # hopefully better? CLAHE whatever that it
    #lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    #l, a, b = cv2.split(lab)
    #clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
    #ab = cv2.merge([clahe.apply(l), a, b])
    #frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  
      
    drawn = alpr.draw_predictions(frame)
    annotated_frame = drawn.image
    results = drawn.results

    out_path = Path(f"../PHOTOS/annotated/{name}")
    out_path = out_path.with_suffix(".jpg")
    cv2.imwrite(str(out_path), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

# make format checking a new function
    if len(results) == 0:
        my_text, my_conf, my_status = "NULL", "< 0.60", "NO_DETECTION"
    else:
        txt_entries = []
        stat_entries = []
        conf_entries = []
        for r in results:
            text, conf = r.ocr.text, np.mean(r.ocr.confidence)
            txt = text if (conf >= 0.90 and text) else "NULL"
            stat = "HIGH" if (conf >= 0.90 and r.ocr.region == "United States") else "LOW" #global dataset, state parsing is complex
            txt_entries.append(txt)
            stat_entries.append(stat)
            conf_entries.append(f"{conf:1.4f}")

        my_text = ' // '.join(txt_entries)
        my_status = ' // '.join(stat_entries)
        my_conf = ' // '.join(conf_entries)

    mismatch = "Y" if my_text != og_ocr else "N"
    loggers["results"].info(f"{name}~{my_text}~{my_conf}~{my_status}~{og_ocr}~{og_status}~{mismatch}")    
    return


def main(loggers):
    og_data = pd.read_csv("./***primary_CSV***", dtype=str, encoding='utf-8-sig')
    og_data.columns = og_data.columns.str.upper()
    og_data['LICENSEPLATE'] = og_data['LICENSEPLATE'].str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
    og_data.fillna("NULL", inplace=True)
    og_lookup = og_data.drop_duplicates(subset='FILENAME', keep='first').set_index('FILENAME')[['LICENSEPLATE', 'STATUS']].to_dict('index')

    trt_providers = [("TensorrtExecutionProvider",
        {"trt_engine_cache_enable": True,
         "trt_engine_cache_path": "/tmp/trt_cache",
         "trt_fp16_enable": True,
        }), "CUDAExecutionProvider", "CPUExecutionProvider"]

    alpr = ALPR(
        detector_model="yolo-v9-t-640-license-plate-end2end",
        detector_conf_thresh=0.6,
        detector_providers=trt_providers,
        ocr_model="cct-s-v2-global-model",
        ocr_providers=trt_providers,
    )

    pics_list = og_data['FILENAME'].str.strip().tolist()
    for file in tqdm(pics_list):
        rel_path = Path(f"../PHOTOS/sheet/{file}")
        if rel_path.name == "NULL":
            loggers["results"].info(f"NULL~NO_NAME")
        elif rel_path.exists() and rel_path.suffix in [".png", ".jpg", ".jpeg", ".webp"]:
            analyze(rel_path, alpr, loggers, og_lookup)
        elif rel_path.exists():
            loggers["results"].info(f"{rel_path.name}~BAD_EXT")
        else:
            loggers["results"].info(f"{rel_path.name}~NOT_FOUND")

    results_df = pd.read_csv("results.csv", sep="~").astype(str)
    results_df.replace({"nan": "NULL", "None": "NULL", "": "NULL"}, inplace=True)
    results_df.fillna("NULL", inplace=True)
    verify_ocr(results_df)
    print_stats(results_df, og_data)
    export_html(results_df)
    return

def print_stats(results_df, og_data, verified_path="verified.csv"):
    total = len(results_df)
    console = Console()

    # load and merge verified if it exists
    try:
        verified = pd.read_csv(verified_path, dtype=str, encoding='utf-8-sig')
        verified.columns = verified.columns.str.upper()
        verified['TRUE'] = verified['TRUE'].str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
        merged_v = results_df.merge(verified, on='FILENAME', how='inner')
        verified_total = len(merged_v)
        verified_mismatches = (merged_v['MY_OCR'] != merged_v['TRUE']).sum()
    except FileNotFoundError:
        merged_v = None
        verified_total = 0
        verified_mismatches = 0

    def pct(n): return f"{n} ({n/total*100:.1f}%)"
    def pct_v(n): return f"{n} ({n/verified_total*100:.1f}%)" if verified_total > 0 else "N/A"

    table = Table(title="Pipeline Stats")
    table.add_column("Metric", style="white")
    table.add_column("Value",  style="cyan")

    stats = [
        ("Total images processed",        str(total)),
        ("Detections (not NO_DETECTION)",  pct((results_df['MY_STATUS'] != 'NO_DETECTION').sum())),
        ("No detections",                  pct((results_df['MY_STATUS'] == 'NO_DETECTION').sum())),
        ("HIGH confidence reads",          pct((results_df['MY_STATUS'] == 'HIGH').sum())),
        ("LOW confidence reads",           pct((results_df['MY_STATUS'] == 'LOW').sum())),
        ("NULL OCR results",               pct((results_df['MY_OCR'] == 'NULL').sum())),
        ("NOT_FOUND / BAD_EXT / errors",   pct(results_df['MY_OCR'].isin(['NOT_FOUND','BAD_EXT','READ_FAILURE']).sum())),
        ("Mismatches vs original CSV",     pct((results_df['MISMATCH'] == 'Y').sum())),
        ("Verified sample size",           str(verified_total)),
        ("Mismatches vs verified truth",   pct_v(verified_mismatches)),
        ("Correct vs verified truth",      pct_v(verified_total - verified_mismatches)),
    ]

    for metric, value in stats:
        table.add_row(metric, value)

    console.print(table)
    return


if __name__ == "__main__":
    start = time.perf_counter()
    loggers, handlers = setup_loggers()
    main(loggers)
    teardown_loggers(handlers)
    print(f"{time.perf_counter()-start:0.2f} sec")
    exit()
