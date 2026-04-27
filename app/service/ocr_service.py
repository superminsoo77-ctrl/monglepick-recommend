"""
영화관 영수증 OCR 서비스 (Tesseract)

영역별(full/top/middle/bottom) 크롭 + 7가지 전처리 variant +
3가지 Tesseract PSM 조합으로 최고 품질 OCR 텍스트를 추출한다.
최종 결과는 [TOP]/[MIDDLE]/[BOTTOM] 태그로 구분해 파서에 전달한다.

보안 정책:
  - SSRF 방어: image_url 은 http/https 만 허용하고, 호스트 해석 결과가 사설/
    루프백/링크로컬/예약 대역이면 거부한다. 운영 환경은 _ALLOWED_HOSTS
    환경변수(쉼표 구분)로 업로드 도메인만 허용하도록 구성할 수 있다.
  - 이미지 크기 제한: 다운로드 스트림을 청크 단위로 누적하며 _MAX_IMAGE_BYTES
    (기본 10MB) 초과 시 즉시 중단하여 메모리 폭발/DoS 를 방지한다.
"""
import io
import os
import re
import socket
import ipaddress
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Tuple, Dict
from urllib.parse import urlparse

import httpx
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 보안 설정 상수
# ──────────────────────────────────────────────

_ALLOWED_SCHEMES: frozenset = frozenset({"http", "https"})
_MAX_IMAGE_BYTES: int = int(os.getenv("OCR_MAX_IMAGE_BYTES", 10 * 1024 * 1024))
_DOWNLOAD_TIMEOUT: float = float(os.getenv("OCR_DOWNLOAD_TIMEOUT", 15.0))
_ALLOWED_HOSTS: frozenset = frozenset(
    h.strip().lower() for h in os.getenv("OCR_ALLOWED_HOSTS", "").split(",") if h.strip()
)

# ──────────────────────────────────────────────
# Tesseract 설정
# ──────────────────────────────────────────────

_TESSERACT_LANG = "kor+eng"
_CONFIGS: List[Tuple[str, str]] = [
    ("psm6",  "--oem 3 --psm 6"),
    ("psm4",  "--oem 3 --psm 4"),
    ("psm11", "--oem 3 --psm 11"),
]

# 짧은 변 기준 최소 해상도 (px)
_MIN_DIM = 800

# 병렬 OCR 워커 수 (환경변수로 조정 가능)
_OCR_MAX_WORKERS: int = int(os.getenv("OCR_MAX_WORKERS", 4))


# ──────────────────────────────────────────────
# 이미지 전처리 유틸
# ──────────────────────────────────────────────

def _resize_to_target(img: Image.Image, min_dim: int = _MIN_DIM) -> Image.Image:
    w, h = img.size
    short = min(w, h)
    if short < min_dim:
        scale = min_dim / short
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _crop_receipt_center(img: Image.Image, margin: float = 0.03) -> Image.Image:
    w, h = img.size
    return img.crop((
        int(w * margin), int(h * margin),
        int(w * (1 - margin)), int(h * (1 - margin)),
    ))


def _crop_regions(img: Image.Image) -> Dict[str, Image.Image]:
    """
    이미지를 4개 관심 영역으로 크롭.
      full   : 전체 이미지
      top    : 상단 0~45%  — 영화관명·날짜·영화명·관람등급
      middle : 중단 35~75% — 좌석·상영관·인원·가격
      bottom : 하단 70~100% — 결제·환불·사업자 정보
    """
    w, h = img.size
    return {
        "full":   img,
        "top":    img.crop((0, 0,           w, int(h * 0.45))),
        "middle": img.crop((0, int(h * 0.35), w, int(h * 0.75))),
        "bottom": img.crop((0, int(h * 0.70), w, h)),
    }


# ──────────────────────────────────────────────
# 이진화 유틸 (OpenCV/NumPy 미사용, 순수 PIL)
# ──────────────────────────────────────────────

def _otsu_threshold_value(gray: Image.Image) -> int:
    """Otsu's method 로 최적 이진화 임계값 계산."""
    hist = gray.histogram()
    total = sum(hist)
    total_sum = sum(i * cnt for i, cnt in enumerate(hist))
    sum_b = weight_b = 0
    max_var = 0.0
    threshold = 127
    for i, cnt in enumerate(hist):
        weight_b += cnt
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += i * cnt
        mean_b = sum_b / weight_b
        mean_f = (total_sum - sum_b) / weight_f
        var = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var > max_var:
            max_var = var
            threshold = i
    return threshold


def _otsu_threshold(gray: Image.Image) -> Image.Image:
    t = _otsu_threshold_value(gray)
    return gray.point(lambda x: 255 if x > t else 0, "L")


def _adaptive_threshold(gray: Image.Image) -> Image.Image:
    """
    Gaussian blur 기반 간이 적응형 이진화.
    각 픽셀을 로컬 평균(블러값)과 비교해 어두운 텍스트를 검정으로 추출.
    """
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=15))
    g_data = gray.getdata()
    b_data = blurred.getdata()
    result = bytes(255 if int(g) > int(b) - 10 else 0 for g, b in zip(g_data, b_data))
    return Image.frombytes("L", gray.size, result)


# ──────────────────────────────────────────────
# 7가지 전처리 variant
# ──────────────────────────────────────────────

def _make_preprocessing_variants(img: Image.Image) -> List[Tuple[str, Image.Image]]:
    """
    (variant_name, 처리된 이미지) 목록 반환.
    모든 이미지는 "L" (grayscale) mode 로 변환된다.
    """
    gray = img.convert("L")
    gray_med = gray.filter(ImageFilter.MedianFilter(size=3))
    auto = ImageOps.autocontrast(gray_med, cutoff=2)

    high = ImageEnhance.Contrast(gray_med).enhance(3.0)
    high = high.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)

    sharp = ImageEnhance.Contrast(gray_med).enhance(2.0)
    sharp = ImageEnhance.Brightness(sharp).enhance(1.1)
    sharp = sharp.filter(ImageFilter.SHARPEN)

    return [
        ("original",           gray),
        ("grayscale",          gray_med),
        ("autocontrast",       auto),
        ("high_contrast",      high),
        ("sharpen",            sharp),
        ("adaptive_threshold", _adaptive_threshold(gray_med)),
        ("otsu_threshold",     _otsu_threshold(gray_med)),
    ]


# ──────────────────────────────────────────────
# OCR 품질 점수 (영역별 가중치 포함)
# ──────────────────────────────────────────────

_BASE_SCORE_PATTERNS: List[Tuple[str, float]] = [
    (r"(?:CGV|메가박스|MEGABOX|롯데\s*시네마|LOTTE\s*CINEMA|B[O0]X\s*KIOSK)", 30.0),
    (r"(?:전체|12세|15세|18세|청소년).{0,4}관람",                               25.0),
    (r"(?:일반|성인|청소년|우대|군인)\s*\d+\s*[명매]",                           20.0),
    (r"\d{4}[./-]\d{1,2}[./-]\d{1,2}",                                        20.0),
    (r"[A-Z가-힣]\s*열\s*\d+\s*번?",                                           20.0),
    (r"\d{2}[./-]\d{1,2}[./-]\d{1,2}",                                        15.0),
    (r"\b[A-Z]\d{1,3}\b",                                                      12.0),
    (r"\d+\s*관(?!람|객)",                                                      12.0),
    (r"(?:영화명|작품명|상영\s*제목)",                                            18.0),
    (r"(?:좌석\s*번호?|SEAT\b)",                                                15.0),
    (r"(?:상영관|관람관|관람일|관람일시|상영일시)",                                 12.0),
    (r"\d{1,2}:\d{2}",                                                         10.0),
    (r"(?:영화|관람|상영|티켓|입장|좌석|KIOSK|발권)",                              8.0),
    (r"\d{1,3}(?:,\d{3})+\s*원",                                                5.0),
]

_AREA_EXTRA_PATTERNS: Dict[str, List[Tuple[str, float]]] = {
    "top": [
        (r"(?:CGV|메가박스|MEGABOX|롯데\s*시네마|LOTTE\s*CINEMA)", 15.0),
        (r"(?:전체|12세|15세|18세|청소년).{0,4}관람",              10.0),
        (r"\d{4}[./-]\d{1,2}[./-]\d{1,2}",                        10.0),
        (r"KIOSK|발권|전체발권|입장권",                              10.0),
        (r"[가-힣]{2,}",                                             3.0),
    ],
    "middle": [
        (r"(?:일반|성인|청소년|우대|군인)\s*\d+\s*[명매]", 10.0),
        (r"[A-Z가-힣]\s*열\s*\d+\s*번?",                  10.0),
        (r"\b[A-Z]\d{1,3}\b",                              6.0),
        (r"\d+\s*관(?!람|객)",                              6.0),
        (r"\d+\s*[명매장]",                                 8.0),
    ],
    "bottom": [
        (r"TOTAL|합계|총금액|총\s*인원",       5.0),
        (r"사업자|환불|카드|매점|부가세",      -10.0),
    ],
    "full": [],
}


def _ocr_score(text: str, area: str = "full") -> float:
    if not text:
        return 0.0
    score = min(len(text) * 0.2, 60.0)
    for pat, boost in _BASE_SCORE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            score += boost
    for pat, boost in _AREA_EXTRA_PATTERNS.get(area, []):
        if re.search(pat, text, re.IGNORECASE):
            score += boost
    # 특수문자 비율이 35% 초과 시 감점
    if len(text) > 10:
        noise = sum(
            1 for c in text
            if not (c.isalnum() or '가' <= c <= '힣' or c in ' \n\t.,:-/()[]　')
        )
        if noise / len(text) > 0.35:
            score -= 20.0
    return score


# ──────────────────────────────────────────────
# 단일 OCR 실행 (ThreadPoolExecutor 용)
# ──────────────────────────────────────────────

def _run_ocr_task(img: Image.Image, lang: str, config: str) -> str:
    try:
        return pytesseract.image_to_string(img, lang=lang, config=config).strip()
    except Exception as e:
        logger.warning("Tesseract 실패 — config=%s error=%s", config, e)
        return ""


# ──────────────────────────────────────────────
# 메인 OCR 실행 — 영역별 × variant × PSM 병렬
# ──────────────────────────────────────────────

def _ocr_multi_region(image_bytes: bytes) -> Tuple[Optional[str], List[str]]:
    """
    4개 영역 × 7 variant × 3 PSM 조합으로 OCR 후
    영역별 최고 점수 텍스트를 [TOP]/[MIDDLE]/[BOTTOM] 태그로 묶어 반환한다.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = _crop_receipt_center(img)
    img = _resize_to_target(img)
    regions = _crop_regions(img)

    # (area, variant, psm_name, variant_img, config) 태스크 목록
    tasks: List[Tuple[str, str, str, Image.Image, str]] = []
    for area_name, area_img in regions.items():
        for var_name, var_img in _make_preprocessing_variants(area_img):
            for psm_name, config in _CONFIGS:
                tasks.append((area_name, var_name, psm_name, var_img, config))

    logger.info("OCR 시작 — 총 %d 조합 (워커=%d)", len(tasks), _OCR_MAX_WORKERS)

    # 병렬 실행
    ocr_results: List[Tuple[str, str, str, str, float]] = []
    with ThreadPoolExecutor(max_workers=_OCR_MAX_WORKERS) as executor:
        future_map = {
            executor.submit(_run_ocr_task, var_img, _TESSERACT_LANG, config):
                (area, var, psm)
            for area, var, psm, var_img, config in tasks
        }
        for future in as_completed(future_map):
            area, var, psm = future_map[future]
            text = future.result()
            score = _ocr_score(text, area)
            ocr_results.append((area, var, psm, text, score))
            logger.debug(
                "OCR %-8s %-22s %-5s score=%6.1f chars=%4d  preview=%s",
                area, var, psm, score, len(text),
                text[:60].replace("\n", " | "),
            )

    # 영역별 최고 점수 텍스트 선택
    area_best: Dict[str, Tuple[str, float, str, str]] = {}
    for area, var, psm, text, score in ocr_results:
        if not text:
            continue
        if area not in area_best or score > area_best[area][1]:
            area_best[area] = (text, score, var, psm)

    # 결과 로깅
    for area in ("full", "top", "middle", "bottom"):
        if area in area_best:
            text, score, var, psm = area_best[area]
            logger.info(
                "최선 %-8s → variant=%-22s psm=%-5s score=%6.1f chars=%4d",
                area, var, psm, score, len(text),
            )
        else:
            logger.warning("최선 %-8s → 결과 없음", area)

    if not area_best:
        return None, []

    # all_texts: 영역별 최고 텍스트 목록 (파서 폴백용)
    all_texts = [info[0] for info in area_best.values() if info[0]]

    top_text    = area_best.get("top",    ("",))[0]
    middle_text = area_best.get("middle", ("",))[0]
    bottom_text = area_best.get("bottom", ("",))[0]
    full_text   = area_best.get("full",   ("",))[0]

    # [TOP]/[MIDDLE]/[BOTTOM] 섹션 결합
    sections: List[str] = []
    if top_text:
        sections.append(f"[TOP]\n{top_text}")
        logger.info("[TOP] best:\n%s", top_text[:300])
    if middle_text:
        sections.append(f"[MIDDLE]\n{middle_text}")
        logger.info("[MIDDLE] best:\n%s", middle_text[:200])
    if bottom_text:
        sections.append(f"[BOTTOM]\n{bottom_text}")
        logger.info("[BOTTOM] best:\n%s", bottom_text[:100])

    if sections:
        combined = "\n".join(sections)
        logger.info("영역 결합 완료 — 총 chars=%d", len(combined))
        return combined, all_texts

    return full_text or None, all_texts


# ──────────────────────────────────────────────
# 보안 — SSRF 방어 + 스트리밍 크기 제한
# ──────────────────────────────────────────────

class UnsafeImageUrlError(ValueError):
    """SSRF 방어에 의해 거부된 URL 입력 오류."""


def _validate_image_url(image_url: str) -> str:
    parsed = urlparse(image_url)

    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeImageUrlError(f"허용되지 않는 스킴: {parsed.scheme!r}")

    host = (parsed.hostname or "").lower()
    if not host:
        raise UnsafeImageUrlError("호스트명이 비어 있습니다")

    if _ALLOWED_HOSTS and host not in _ALLOWED_HOSTS:
        raise UnsafeImageUrlError(f"허용되지 않은 호스트: {host!r}")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise UnsafeImageUrlError(f"호스트 해석 실패: {host!r} ({e})") from e

    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise UnsafeImageUrlError(
                f"내부/예약 IP 대역은 허용되지 않습니다: {host!r} → {ip_str}"
            )

    return image_url


async def _download_image_bytes(image_url: str) -> Optional[bytes]:
    try:
        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=False) as client:
            async with client.stream("GET", image_url) as response:
                response.raise_for_status()

                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        if int(content_length) > _MAX_IMAGE_BYTES:
                            logger.warning(
                                "이미지 크기 초과(선언) — Content-Length=%s limit=%d",
                                content_length, _MAX_IMAGE_BYTES,
                            )
                            return None
                    except ValueError:
                        pass

                buf = bytearray()
                async for chunk in response.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > _MAX_IMAGE_BYTES:
                        logger.warning(
                            "이미지 크기 초과(스트림) — 누적=%d limit=%d",
                            len(buf), _MAX_IMAGE_BYTES,
                        )
                        return None
                return bytes(buf)
    except httpx.HTTPError as e:
        logger.error("이미지 다운로드 실패 url=%s error=%s", image_url, e)
        return None


# ──────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────

def re_search_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


async def extract_text_from_url(image_url: str) -> Tuple[Optional[str], List[str]]:
    try:
        _validate_image_url(image_url)
    except UnsafeImageUrlError as e:
        logger.warning("OCR URL 거부 — %s", e)
        return None, []

    image_bytes = await _download_image_bytes(image_url)
    if not image_bytes:
        return None, []

    try:
        best_text, all_texts = _ocr_multi_region(image_bytes)
        if not best_text:
            logger.warning("OCR 추출 결과 없음")
            return None, []
        logger.info("OCR 추출 완료 — 글자 수: %d", len(best_text))
        return best_text, all_texts
    except Exception as e:
        logger.error("OCR 처리 오류: %s", e)
        return None, []