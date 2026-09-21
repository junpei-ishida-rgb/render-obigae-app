# -*- coding: utf-8 -*-
import base64
import io
import os
import uuid
import zipfile
from datetime import datetime, timedelta

from flask import Flask, render_template, request, send_file, abort, jsonify

import band_logic

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

# 処理結果を一時的に保持するインメモリストア（シンプルな社内ツール想定）
# キー: job_id, 値: { created_at, results: [...] }
JOBS = {}
JOB_TTL_MINUTES = 60


def cleanup_old_jobs():
    now = datetime.utcnow()
    expired = [jid for jid, j in JOBS.items()
               if now - j["created_at"] > timedelta(minutes=JOB_TTL_MINUTES)]
    for jid in expired:
        JOBS.pop(jid, None)


STATUS_LABEL = {
    "high": ("自動判定：問題なし", "ok"),
    "medium": ("自動判定：念のため確認推奨", "warn"),
    "low": ("要確認：自動判定に自信なし", "danger"),
    "failed": ("失敗：手動で対応してください", "danger"),
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process_api", methods=["POST"])
def process_api():
    """フロント(JS)からアップロードし、結果一式（サムネ＋ダウンロード用データ）を
    JSONで返すAPI。index.html はこちらを使う。"""
    cleanup_old_jobs()

    files = request.files.getlist("pdf_files")
    if not files:
        return jsonify({"error": "PDFファイルを選択してください。"}), 400

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"created_at": datetime.utcnow(), "files": {}}

    response_files = []

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        data = f.read()
        try:
            result = band_logic.process_pdf(data, f.filename)
        except Exception as e:
            response_files.append({
                "filename": f.filename,
                "error": f"処理に失敗しました: {e}",
            })
            continue

        file_key = str(uuid.uuid4())
        JOBS[job_id]["files"][file_key] = {
            "filename": result["filename"],
            "combined_pdf_bytes": result["combined_pdf_bytes"],
            "pages": result["pages"],  # 生バイト列も保持（個別DL用）
        }

        pages_view = []
        for p in result["pages"]:
            label, css_class = STATUS_LABEL.get(p["status"], ("不明", "warn"))
            pages_view.append({
                "page_index": p["page_index"],
                "status": p["status"],
                "status_label": label,
                "status_class": css_class,
                "thumb_b64": base64.b64encode(p["thumbnail_png"]).decode("ascii"),
                "before_thumb_b64": base64.b64encode(p["before_thumbnail_png"]).decode("ascii"),
            })

        response_files.append({
            "filename": result["filename"],
            "n_pages": result["n_pages"],
            "file_key": file_key,
            "pages": pages_view,
        })

    return jsonify({"job_id": job_id, "files": response_files})


@app.route("/download/<job_id>/<file_key>/combined", methods=["GET"])
def download_combined(job_id, file_key):
    job = JOBS.get(job_id)
    if not job or file_key not in job["files"]:
        abort(404)
    entry = job["files"][file_key]
    buf = io.BytesIO(entry["combined_pdf_bytes"])
    out_name = os.path.splitext(entry["filename"])[0] + "_帯替え後.pdf"
    return send_file(buf, mimetype="application/pdf",
                      as_attachment=True, download_name=out_name)


@app.route("/download/<job_id>/<file_key>/page/<int:page_index>", methods=["GET"])
def download_page(job_id, file_key, page_index):
    job = JOBS.get(job_id)
    if not job or file_key not in job["files"]:
        abort(404)
    entry = job["files"][file_key]
    pages = entry["pages"]
    if page_index < 0 or page_index >= len(pages):
        abort(404)
    buf = io.BytesIO(pages[page_index]["pdf_bytes"])
    base = os.path.splitext(entry["filename"])[0]
    out_name = f"{base}_p{page_index+1}_帯替え後.pdf"
    return send_file(buf, mimetype="application/pdf",
                      as_attachment=True, download_name=out_name)


@app.route("/download/<job_id>/<file_key>/zip", methods=["GET"])
def download_zip(job_id, file_key):
    job = JOBS.get(job_id)
    if not job or file_key not in job["files"]:
        abort(404)
    entry = job["files"][file_key]
    base = os.path.splitext(entry["filename"])[0]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in entry["pages"]:
            name = f"{base}_p{p['page_index']+1}_帯替え後.pdf"
            zf.writestr(name, p["pdf_bytes"])
    buf.seek(0)
    return send_file(buf, mimetype="application/zip",
                      as_attachment=True, download_name=f"{base}_帯替え後_ページ別.zip")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
