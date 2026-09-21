# -*- coding: utf-8 -*-
"""
帯替え自動化コアロジック

処理の考え方:
  1. 入力PDFを1ページずつ分解する（レインズの一括DLなど、複数社の図面が
     1つのPDFに束ねられているケースがあるため）
  2. 各ページについて、テキストが十分に抽出できるか（＝ベクターPDFか）を
     判定する
  3-a. ベクターPDFの場合：
       ページ下部にある「帯（会社情報ボックス）」の境界を、
       - 罫線ボックス（get_drawings）のうち下部25%にあり幅が広いもの
       - それが見つからない場合はアンカー文字列（TEL/FAX/株式会社等）
       のいずれかで検出し、その領域を白塗り（redaction）した上で
       自社の帯PDFをベクターのまま重ね込む（画質劣化なし）
  3-b. ラスターPDF（画像PDF）の場合、またはベクター判定に失敗した場合：
       ページ全体を画像としてレンダリングし、下部の罫線（実線・破線とも
       対応）を検出して帯の境界を特定。境界より上を保持し、下に自社の
       帯を重ね込む
  4. 検出に自信が持てないページは「要確認」として警告リストに積む
     （自動判定を過信せず、必ず人の目でチェックできるようにするため）
"""

import io
import os
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

BAND_CONFIG = {
    "yoko": {
        "path": os.path.join(ASSETS_DIR, "amuse_band_yoko.pdf"),
        "clip": fitz.Rect(22.86, 453.66, 762.42, 523.02),
    },
    "tate": {
        "path": os.path.join(ASSETS_DIR, "amuse_band_tate.pdf"),
        "clip": fitz.Rect(16.38, 712.38, 528.30, 760.50),
    },
}

# ベクターPDFで「会社情報ボックス」を判定するための手がかり文字列
ANCHOR_KEYWORDS = [
    "TEL", "FAX", "株式会社", "㈱", "有限会社",
    "国土交通大臣", "東京都知事", "immo", "免許",
]

# 十分なテキストがないページは「ラスター扱い」にする文字数しきい値
TEXT_LEN_THRESHOLD = 150


def get_band_asset(orientation: str):
    cfg = BAND_CONFIG[orientation]
    doc = fitz.open(cfg["path"])
    clip = cfg["clip"]
    return doc, clip


def choose_orientation(page_rect: fitz.Rect) -> str:
    """ページの縦横比から、横帯・縦帯どちらを使うか自動判定する。"""
    return "yoko" if page_rect.width >= page_rect.height else "tate"


def fit_rect(target: fitz.Rect, src_w: float, src_h: float, align="bottom-left") -> fitz.Rect:
    """target矩形の中に、アスペクト比を保ったまま src_w x src_h を収める
    矩形を計算する（帯が歪まないようにするため）。"""
    target_w, target_h = target.width, target.height
    src_aspect = src_w / src_h
    target_aspect = target_w / target_h

    if target_aspect > src_aspect:
        # target の方が横長 → 高さいっぱいに合わせる
        new_h = target_h
        new_w = new_h * src_aspect
    else:
        # target の方が縦長（または帯の方が横長） → 幅いっぱいに合わせる
        new_w = target_w
        new_h = new_w / src_aspect

    if align == "bottom-left":
        x0 = target.x0
        y1 = target.y1
        y0 = y1 - new_h
        x1 = x0 + new_w
    else:  # center
        cx = (target.x0 + target.x1) / 2
        cy = (target.y0 + target.y1) / 2
        x0 = cx - new_w / 2
        x1 = cx + new_w / 2
        y0 = cy - new_h / 2
        y1 = cy + new_h / 2

    return fitz.Rect(x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# ベクターPDF: 帯（会社情報ボックス）の検出
# ---------------------------------------------------------------------------

def find_footer_box_by_border(page: fitz.Page):
    """下部にある「幅広の罫線ボックス」を帯の候補として探す。
    見つかれば (Rect, confidence='high') を返す。"""
    rect_h = page.rect.height
    rect_w = page.rect.width
    candidates = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.width > rect_w * 0.45 and 15 < r.height < rect_h * 0.35 and r.y0 > rect_h * 0.55:
            candidates.append(r)
    if not candidates:
        return None, None
    # 一番下にある／一番大きいものを採用
    best = max(candidates, key=lambda r: r.width * r.height)
    return fitz.Rect(best), "high"


def find_footer_box_by_image(page: fitz.Page):
    """会社ロゴ・住所・TEL等が1枚の画像として埋め込まれているタイプ
    （例: 会社名で検索してもヒットしないテンプレート）向け。
    下部にある幅広の画像を帯の候補として探す。"""
    rect_h = page.rect.height
    rect_w = page.rect.width
    candidates = []
    for info in page.get_image_info(xrefs=True):
        bbox = fitz.Rect(info["bbox"])
        if (bbox.width > rect_w * 0.15
                and 10 < bbox.height < rect_h * 0.35
                and bbox.y0 > rect_h * 0.55
                and bbox.x0 < rect_w * 0.65):
            candidates.append(bbox)
    if not candidates:
        return None, None
    best = max(candidates, key=lambda b: b.width * b.height)
    return best, "high"


def find_footer_box_by_anchor(page: fitz.Page):
    """罫線ボックスが見つからない場合のフォールバック：
    TEL/FAX/株式会社などのアンカー文字列の位置から帯の範囲を推定する。"""
    rect_h = page.rect.height
    rect_w = page.rect.width
    hits = []
    for kw in ANCHOR_KEYWORDS:
        for r in page.search_for(kw):
            if r.y0 > rect_h * 0.55:
                hits.append(r)
    if not hits:
        return None, None

    min_y0 = min(r.y0 for r in hits)
    # 少し上に余白を持たせる
    pad = 6
    box = fitz.Rect(0, max(0, min_y0 - pad), rect_w, rect_h)
    return box, "low"


def replace_band_vector(doc: fitz.Document, page: fitz.Page, band_doc, band_clip):
    # 優先順位: ①会社情報が画像として埋め込まれているケース
    #          ②罫線で囲まれた帯ボックスが見つかるケース
    #          ③アンカー文字列（TEL/FAX等）からの推定（フォールバック）
    box, confidence = find_footer_box_by_image(page)
    if box is None:
        box, confidence = find_footer_box_by_border(page)
    if box is None:
        box, confidence = find_footer_box_by_anchor(page)
    if box is None:
        return "failed", None

    page.add_redact_annot(box, fill=(1, 1, 1))
    page.apply_redactions()

    band_w = band_clip.width
    band_h = band_clip.height
    fitted = fit_rect(box, band_w, band_h, align="bottom-left")
    page.show_pdf_page(fitted, band_doc, 0, clip=band_clip)
    return confidence, box


# ---------------------------------------------------------------------------
# ラスターPDF（画像ページ）: 帯境界の検出
# ---------------------------------------------------------------------------

def find_band_top_row(pil_img: Image.Image, search_from_frac=0.55, search_to_frac=0.97):
    """画像を下から上にスキャンし、帯の上端となる横罫線（実線・破線とも）
    を検出する。見つかった場合 (y, confidence) を返す。
    ページ最下端ギリギリ（外枠の罫線など）を誤検出しないよう、
    search_to_frac で下端付近を検索対象から除外する。"""
    arr = np.array(pil_img.convert("L"))
    h, w = arr.shape
    start_y = int(h * search_from_frac)
    end_y = int(h * search_to_frac)  # これより下（ページ最下端寄り）は見ない

    # 1st pass: 実線（行の80%以上が暗い）
    for y in range(end_y, start_y, -1):
        row = arr[y]
        if (row < 150).mean() > 0.8:
            return y, "high"

    # 2nd pass: 破線・点線（暗いピクセルは少ないが、周期的に幅広く分布）
    for y in range(end_y, start_y, -1):
        row = arr[y]
        dark = row < 150
        dark_frac = dark.mean()
        if 0.15 < dark_frac < 0.8:
            # 暗い部分が画像の左端から右端まで広く分布しているかチェック
            dark_idx = np.where(dark)[0]
            spread = (dark_idx.max() - dark_idx.min()) / w if len(dark_idx) else 0
            if spread > 0.7:
                return y, "medium"

    return None, None


def replace_band_raster(page: fitz.Page, band_doc, band_clip, dpi=200):
    """ページを画像としてレンダリングし、帯を検出して差し替えた
    新しい1ページPDFを返す。"""
    pix = page.get_pixmap(dpi=dpi)
    pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    band_top_px, confidence = find_band_top_row(pil_img)
    if band_top_px is None:
        # 検出に失敗した場合は既定値（88%位置）を使い、低信頼度として扱う
        band_top_px = int(pil_img.height * 0.88)
        confidence = "low"

    top_crop = pil_img.crop((0, 0, pil_img.width, band_top_px))

    page_w = page.rect.width
    content_h_pt = page_w * (top_crop.height / top_crop.width)

    band_w = band_clip.width
    band_h = band_clip.height
    band_h_pt = page_w * (band_h / band_w)

    new_doc = fitz.open()
    new_page = new_doc.new_page(width=page_w, height=content_h_pt + band_h_pt)

    buf = io.BytesIO()
    top_crop.save(buf, format="PNG")
    new_page.insert_image(fitz.Rect(0, 0, page_w, content_h_pt), stream=buf.getvalue())

    band_rect = fitz.Rect(0, content_h_pt, page_w, content_h_pt + band_h_pt)
    new_page.show_pdf_page(band_rect, band_doc, 0, clip=band_clip)

    return new_doc, confidence


# ---------------------------------------------------------------------------
# ページ単位の処理エントリーポイント
# ---------------------------------------------------------------------------

def process_single_page(src_doc: fitz.Document, page_index: int):
    """1ページを処理し、(出力用1ページPDFのbytes, ステータス) を返す。
    ステータス: 'high' | 'medium' | 'low' | 'failed'
    """
    page = src_doc[page_index]
    orientation = choose_orientation(page.rect)
    band_doc, band_clip = get_band_asset(orientation)

    text_len = len(page.get_text())

    if text_len >= TEXT_LEN_THRESHOLD:
        # ベクターPDFとして処理するため、ページ単体を複製して作業する
        tmp_doc = fitz.open()
        tmp_doc.insert_pdf(src_doc, from_page=page_index, to_page=page_index)
        tmp_page = tmp_doc[0]
        status, _box = replace_band_vector(tmp_doc, tmp_page, band_doc, band_clip)
        if status == "failed":
            # ベクター検出に失敗 → ラスター方式にフォールバック
            band_doc2, band_clip2 = get_band_asset(orientation)
            new_doc, status = replace_band_raster(page, band_doc2, band_clip2)
            out_bytes = new_doc.tobytes()
            new_doc.close()
            band_doc2.close()
        else:
            out_bytes = tmp_doc.tobytes()
        tmp_doc.close()
    else:
        new_doc, status = replace_band_raster(page, band_doc, band_clip)
        out_bytes = new_doc.tobytes()
        new_doc.close()

    band_doc.close()
    return out_bytes, status


def render_thumbnail(pdf_bytes: bytes, dpi=90):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[0].get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


def process_pdf(file_bytes: bytes, original_filename: str):
    """複数ページのPDFを処理し、ページごとの結果リストと、
    全ページ結合済みの出力PDFを返す。"""
    src_doc = fitz.open(stream=file_bytes, filetype="pdf")
    n_pages = len(src_doc)

    results = []
    combined = fitz.open()

    for i in range(n_pages):
        # 差し替え前のサムネイル（見比べ用）
        before_pix = src_doc[i].get_pixmap(dpi=90)
        before_thumb = before_pix.tobytes("png")

        out_bytes, status = process_single_page(src_doc, i)
        after_thumb = render_thumbnail(out_bytes)

        page_doc = fitz.open(stream=out_bytes, filetype="pdf")
        combined.insert_pdf(page_doc)
        page_doc.close()

        results.append({
            "page_index": i,
            "status": status,
            "pdf_bytes": out_bytes,
            "thumbnail_png": after_thumb,
            "before_thumbnail_png": before_thumb,
        })

    combined_bytes = combined.tobytes()
    combined.close()
    src_doc.close()

    return {
        "filename": original_filename,
        "n_pages": n_pages,
        "pages": results,
        "combined_pdf_bytes": combined_bytes,
    }
