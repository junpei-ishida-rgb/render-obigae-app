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

# ベクターPDFで「会社情報・取引条件ボックス」を判定するための手がかり文字列
# （お客様に見せてはいけない業者間情報＝仲介手数料・取引態様なども含めて
#   検出できるよう拾うが、通常の物件案内文にも出てきがちな一般的すぎる
#   語（「仲介手数料」「仲介業者様」など単体）は誤検出の元になるため避け、
#   なるべく footer 特有の言い回しに絞る）
ANCHOR_KEYWORDS = [
    "TEL", "FAX", "株式会社", "㈱", "有限会社",
    "国土交通大臣", "東京都知事", "免許",
    "取引態様", "取引形態",
    "客付", "元付", "AD：", "AD:", "広告料",
    "仲介業者様専用", "社宅利用", "鍵：現地対応", "ITANDI",
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

def find_footer_top_candidates(page: fitz.Page):
    """帯・取引条件欄を構成しうる要素（画像／罫線ボックス／キーワード文字列）
    をすべて集め、それぞれの上端y座標と信頼度ラベルを返す。
    ここで集めた候補のうち一番上（最小y0）を「帯の開始位置」として採用し、
    そこから下をページ全幅で白塗りする（項目ごとに個別の矩形は使わない）。
    """
    rect_h = page.rect.height
    rect_w = page.rect.width
    # 罫線ボックス／画像は形状の条件が厳しいので広めの範囲で探して良いが、
    # キーワード検索は本文中に同じ語（取引態様・貸主等）が別の意味で出て
    # くることがあるため、より下部（ページ最下部付近）に絞って誤検出を防ぐ
    search_top_shape = rect_h * 0.55
    search_top_keyword = rect_h * 0.80
    candidates = []  # (y0, confidence)

    # ① 罫線で囲まれた幅広ボックス
    for d in page.get_drawings():
        r = d["rect"]
        if r.width > rect_w * 0.45 and 15 < r.height < rect_h * 0.35 and r.y0 > search_top_shape:
            candidates.append((r.y0, "high"))

    # ② 会社ロゴ等が画像として埋め込まれているケース
    for info in page.get_image_info(xrefs=True):
        bbox = fitz.Rect(info["bbox"])
        if (bbox.width > rect_w * 0.15
                and 10 < bbox.height < rect_h * 0.35
                and bbox.y0 > search_top_shape
                and bbox.x0 < rect_w * 0.65):
            candidates.append((bbox.y0, "high"))

    # ③ アンカー文字列（TEL/FAX/株式会社/取引態様/仲介手数料 等）
    for kw in ANCHOR_KEYWORDS:
        for r in page.search_for(kw):
            if r.y0 > search_top_keyword:
                candidates.append((r.y0, "medium"))

    return candidates


def replace_band_vector(doc: fitz.Document, page: fitz.Page, band_doc, band_clip):
    """帯・取引条件欄の中で一番上にある要素を探し、そこから下をページ全幅で
    白塗りしたうえで、自社の帯を全幅で重ね込む。
    （物件の設備欄などがその下端より下にはみ出ていない前提。実務上、
    帯・取引条件欄はページの一番下にまとまっているため、この前提で問題ない）
    """
    rect_h = page.rect.height
    rect_w = page.rect.width

    candidates = find_footer_top_candidates(page)
    if not candidates:
        return "failed", None

    min_y0 = min(y for y, _c in candidates)
    # 一番上の候補がどの検出方法由来かで信頼度を決める
    confidence = "high" if any(y == min_y0 and c == "high" for y, c in candidates) else "medium"

    pad = 4
    box = fitz.Rect(0, max(0, min_y0 - pad), rect_w, rect_h)

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
    """画像を下から上にスキャンし、帯の上端となる横罫線を検出する。
    以下3種類の線に対応する:
      - 実線（黒・グレー系の暗い罫線）
      - 破線・点線
      - 色付きの帯（オレンジ色の区切りバー等、暗くはないが単色でページ幅
        いっぱいに広がっている線）
    見つかった場合 (y, confidence) を返す。
    ページ最下端ギリギリ（外枠の罫線など）を誤検出しないよう、
    search_to_frac で下端付近を検索対象から除外する。"""
    gray = np.array(pil_img.convert("L"))
    rgb = np.array(pil_img.convert("RGB"))
    h, w = gray.shape
    start_y = int(h * search_from_frac)
    end_y = int(h * search_to_frac)  # これより下（ページ最下端寄り）は見ない

    for y in range(end_y, start_y, -1):
        row_gray = gray[y]

        # ①実線（行の80%以上が暗い）
        if (row_gray < 150).mean() > 0.8:
            return y, "high"

        # ②単色の帯（色は問わない。行の85%以上が同じ色で、かつ白に近すぎない）
        row_rgb = rgb[y]
        bucketed = (row_rgb // 24).astype(np.int32)
        keys = bucketed[:, 0] * 10000 + bucketed[:, 1] * 100 + bucketed[:, 2]
        vals, counts = np.unique(keys, return_counts=True)
        top_idx = counts.argmax()
        frac = counts[top_idx] / len(row_rgb)
        if frac > 0.85:
            key = vals[top_idx]
            r, g, b = key // 10000, (key // 100) % 100, key % 100
            brightness = (r + g + b) / 3 * 24
            if brightness < 235:
                return y, "high"

        # ③破線・点線（暗いピクセルは少ないが、均等な間隔で幅広く分布する
        #   小さな線分の繰り返しになっている）
        dark = row_gray < 150
        dark_frac = dark.mean()
        if 0.15 < dark_frac < 0.8:
            dark_idx = np.where(dark)[0]
            spread = (dark_idx.max() - dark_idx.min()) / w if len(dark_idx) else 0
            if spread > 0.85:
                # 連続した暗い区間（セグメント）に分割し、本文テキストの
                # 行（大きさや間隔がバラバラ）と区別する
                gaps = np.where(np.diff(dark_idx) > 1)[0]
                segments = np.split(dark_idx, gaps + 1)
                seg_lengths = [len(s) for s in segments]
                if len(segments) >= 12:
                    max_len = max(seg_lengths)
                    min_len = min(seg_lengths)
                    if max_len <= w * 0.02 and max_len / max(min_len, 1) < 4:
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
