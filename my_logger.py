from rich.console import Console
from rich.table import Table
import logging
import pandas as pd
import re

def setup_loggers():
    loggers = {}
    handlers = {}

    configs = {
      #  "green": "green.html",
      #  "yellow": "yellow.html",
      #  "red": "red.html",
        "results": "results.csv",
    }

    for name, filepath in configs.items():
        logger = logging.getLogger(name)
        handler = logging.FileHandler(filepath, mode="w")
        level = logging.INFO if name == "results" else logging.WARNING
        logger.setLevel(level)
        logger.addHandler(handler)
        logger.propagate = False
        loggers[name] = logger
        handlers[name] = handler

    loggers["results"].info("FILENAME~MY_OCR~MY_CONF~MY_STATUS~OG_OCR~OG_STATUS~MISMATCH")
    return loggers, handlers

def teardown_loggers(handlers):
    for h in handlers.values():
        h.close()
    return

def export_html(df):
    def color_status(val):
        colors = {
            "HIGH":         "background-color: #2d6a2d; color: white",
            "LOW":          "background-color: #7a7a00; color: white",
            "EMPTY":        "background-color: #8b0000; color: white",
            "WRONG_REGION": "background-color: #8b0000; color: white",
            "NO_DETECTION": "background-color: #8b0000; color: white",
            "NULL":         "background-color: #444444; color: white",
        }
        return colors.get(val, "")

    def color_mismatch(val):
        return "background-color: #8b0000; color: white" if val == "Y" else ""

    styled = (
        df.style
        .map(color_status, subset=["MY_STATUS"])
        .map(color_mismatch, subset=["MISMATCH"])
    )

    prefix = "file://///wsl.localhost/***annotated_dir***"
    df['FILENAME'] = df['FILENAME'].apply(lambda x: f'<a href="{prefix}{x}">{x}</a>')
    styled.to_html("results_review.html", mode="w", max_rows=None)
    return

def mismatch_table(df):
    table = Table(title="OCR Mismatches")
    
    table.add_column("Filename",   style="white")
    table.add_column("My OCR",     style="green")
    table.add_column("OG OCR",     style="purple")

    mismatches = df[df["MISMATCH"] == "Y"]
    
    for _, row in mismatches.iterrows():
        table.add_row(row["FILENAME"], row["MY_OCR"], row["OG_OCR"])

    console = Console(record=True)
    console.print(table)
    console.save_svg("mismatches.svg")
    return

def verify_ocr(results_df):
    verified = pd.read_csv("./verified.csv", dtype=str, encoding='utf-8-sig')
    verified['TRUE'] = verified['TRUE'].str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)

    merged = results_df.merge(verified, on='FILENAME', how='left')
    mismatches = merged[merged['MY_OCR'] != merged['TRUE']][['FILENAME', 'MY_OCR', 'TRUE']]
    mismatches.to_csv("ocr_vs_verified.csv", mode="w", index=False)
    return