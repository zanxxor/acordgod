# -*- coding: utf-8 -*-
"""
AcordGod - Extractor de acordes desde partituras PDF.

Funciona con PDFs de "acordes y letra" tipo songbook, donde la línea de
acordes va arriba de la línea de letra, alineados por columna:

    G          D       Em
    Cuán grande es Él, mi alma canta

Requiere que el PDF tenga texto seleccionable (no una foto/escaneo).
"""
import re

import pdfplumber

# Acordes en inglés y español (con sostenido/bemol y sufijos comunes)
_RAIZ_ES = r"(?:Do|Re|Mi|Fa|Sol|La|Si)"
_RAIZ_EN = r"[A-G]"
_ALT = r"[#b♯♭]?"
_SUF = r"(?:maj7|maj9|m7b5|sus2|sus4|add9|dim7|m7|m9|m6|6|7|9|11|13|dim|aug|m|°|\+)?"
_BAJO = r"(?:/(?:Do|Re|Mi|Fa|Sol|La|Si|[A-G])[#b♯♭]?)?"
CHORD_RE = re.compile(rf"^(?:{_RAIZ_ES}|{_RAIZ_EN}){_ALT}{_SUF}{_BAJO}$")


def _es_linea_de_acordes(palabras: list[dict]) -> bool:
    """Una línea es 'de acordes' si la mayoría de sus tokens parecen acordes."""
    if not palabras:
        return False
    validos = sum(1 for p in palabras if CHORD_RE.match(p["text"]))
    return validos >= max(1, len(palabras)) * 0.6


def _agrupar_lineas(words: list[dict], tol: float = 3.0) -> list[list[dict]]:
    """Agrupa palabras de pdfplumber en líneas según su coordenada Y."""
    lineas: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        colocado = False
        for ln in lineas:
            if abs(ln[0]["top"] - w["top"]) <= tol:
                ln.append(w)
                colocado = True
                break
        if not colocado:
            lineas.append([w])
    return lineas


def _insertar_acordes(letra: str, x0_letra: float, acordes: list[dict]) -> str:
    """Inserta cada acorde en la posición de la letra más cercana a su x0."""
    if not acordes:
        return letra
    out, ultimo_pos = [], 0
    # tamaño de carácter aproximado a partir del ancho de la línea de letra
    ancho_total = max((a["x1"] for a in acordes), default=x0_letra + 1) - x0_letra
    ancho_total = max(ancho_total, len(letra) * 6, 1)
    px_por_char = ancho_total / max(len(letra), 1)
    for ac in sorted(acordes, key=lambda a: a["x0"]):
        col = int(round((ac["x0"] - x0_letra) / px_por_char))
        col = max(ultimo_pos, min(col, len(letra)))
        out.append(letra[ultimo_pos:col])
        out.append(f"[{ac['text']}]")
        ultimo_pos = col
    out.append(letra[ultimo_pos:])
    return "".join(out)


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ#b♯♭/]+")


def extraer_secuencia(ruta_pdf: str) -> dict:
    """Para PDFs de diagramas de acordes (sin letra), extrae la progresión
    de acordes (en orden, sin repetidos consecutivos) y el capo si aparece."""
    acordes: list[str] = []
    capo = None
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            texto = pagina.extract_text() or ""
            if capo is None:
                m = re.search(r"capo\s*(?:en\s*el\s*traste\s*)?(\d+)", texto, re.IGNORECASE)
                if m:
                    capo = int(m.group(1))
            for linea in texto.splitlines():
                if "(cid:" in linea:
                    continue
                for tok in _TOKEN_RE.findall(linea):
                    if CHORD_RE.match(tok):
                        if not acordes or acordes[-1] != tok:
                            acordes.append(tok)
    return {"acordes": acordes, "capo": capo}


def es_solo_diagramas(cuerpo: str) -> bool:
    """True si el texto extraído es mayormente glifos sin mapear (cid:N),
    típico de diagramas de digitación sin letra real."""
    sin_marcas = re.sub(r"\[[^\]]*\]", "", cuerpo)
    sin_marcas = re.sub(r"\(cid:\d+\)", "", sin_marcas)
    palabras = re.findall(r"[A-Za-zÀ-ÿ]{2,}", sin_marcas)
    return len(palabras) < 30


def extraer(ruta_pdf: str) -> str:
    """Devuelve el cuerpo en formato AcordGod (letra con [Acordes]) extraído del PDF."""
    salida: list[str] = []
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            words = pagina.extract_words(use_text_flow=False, keep_blank_chars=False)
            lineas = _agrupar_lineas(words)
            i = 0
            while i < len(lineas):
                ln = lineas[i]
                texto = " ".join(p["text"] for p in ln)
                if _es_linea_de_acordes(ln):
                    siguiente = lineas[i + 1] if i + 1 < len(lineas) else None
                    if siguiente and not _es_linea_de_acordes(siguiente):
                        letra = " ".join(p["text"] for p in siguiente)
                        x0 = min(p["x0"] for p in siguiente)
                        salida.append(_insertar_acordes(letra, x0, ln))
                        i += 2
                        continue
                    # línea de solo acordes (instrumental)
                    salida.append(" ".join(f"[{p['text']}]" for p in ln))
                    i += 1
                    continue
                salida.append(texto)
                i += 1
            salida.append("")  # separación entre páginas
    # colapsa líneas en blanco repetidas
    texto = "\n".join(salida)
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip("\n")
    return texto
