# -*- coding: utf-8 -*-
"""
AcordGod - Transcriptor automático de alabanzas.

Uso:
    python transcribir.py "https://www.youtube.com/watch?v=XXXX" [--titulo "Nombre"]

Descarga el audio del video de YouTube, transcribe la letra con Whisper
(con marcas de tiempo) y detecta los acordes con análisis de croma.
Genera:
  - <titulo>.txt            -> letra con acordes en formato AcordGod/ChordPro
  - <titulo>.acordgod.json  -> archivo importable directamente en la app
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SALIDA = HERE / "salida"

NOTAS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# ---------------------------------------------------------------- descarga
def descargar_audio(url: str) -> tuple[Path, str]:
    """Descarga el audio del video y devuelve (ruta_wav, titulo_video)."""
    import yt_dlp

    SALIDA.mkdir(exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(SALIDA / "audio_tmp.%(ext)s"),
        "overwrites": True,
        "quiet": True,
        "noprogress": False,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
        ],
    }
    print(">> Descargando audio de YouTube...")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    titulo = info.get("title", "alabanza")
    return SALIDA / "audio_tmp.wav", titulo


# ------------------------------------------------------------- letra (IA)
def transcribir_letra(wav: Path) -> list[dict]:
    """Transcribe la letra con Whisper. Devuelve segmentos con palabras y tiempos."""
    from faster_whisper import WhisperModel

    print(">> Transcribiendo la letra (Whisper, puede tardar varios minutos la primera vez)...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(wav), language="es", word_timestamps=True, vad_filter=True
    )
    resultado = []
    for seg in segments:
        palabras = [
            {"w": w.word.strip(), "t": float(w.start)}
            for w in (seg.words or [])
            if w.word.strip()
        ]
        if palabras:
            resultado.append({"inicio": float(seg.start), "palabras": palabras})
            print(f"   {seg.start:6.1f}s  {seg.text.strip()}")
    return resultado


# --------------------------------------------------------------- acordes
def detectar_acordes(wav: Path) -> list[tuple[float, str]]:
    """Detecta acordes mayores/menores por análisis de croma sincronizado a beats.

    Devuelve lista de (tiempo_inicio, acorde) solo en los cambios de acorde.
    """
    import librosa
    import scipy.ndimage

    print(">> Detectando acordes...")
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    y_harm = librosa.effects.harmonic(y, margin=4)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma = scipy.ndimage.median_filter(chroma, size=(1, 9))

    # croma promedio por beat
    if len(beats) < 4:
        frames = np.arange(0, chroma.shape[1], 43)  # ~1 s
    else:
        frames = beats
    sync = librosa.util.sync(chroma, frames, aggregate=np.median)
    tiempos = librosa.frames_to_time(frames, sr=sr)

    # plantillas de acordes mayores y menores
    plantillas, nombres = [], []
    for i, nota in enumerate(NOTAS):
        maj = np.zeros(12); maj[[i, (i + 4) % 12, (i + 7) % 12]] = 1
        minr = np.zeros(12); minr[[i, (i + 3) % 12, (i + 7) % 12]] = 1
        plantillas += [maj, minr]
        nombres += [nota, nota + "m"]
    plantillas = np.array(plantillas)
    plantillas = plantillas / np.linalg.norm(plantillas, axis=1, keepdims=True)

    normas = np.linalg.norm(sync, axis=0, keepdims=True)
    normas[normas == 0] = 1
    similitud = plantillas @ (sync / normas)
    indices = np.argmax(similitud, axis=0)

    # suavizado: ventana de mayoría para evitar parpadeo de acordes
    suaves = scipy.ndimage.median_filter(indices, size=5)

    cambios = []
    anterior = None
    for t, idx in zip(tiempos, suaves):
        nombre = nombres[int(idx)]
        if nombre != anterior:
            cambios.append((float(t), nombre))
            anterior = nombre

    # descarta acordes demasiado cortos (<1 s): suelen ser ruido del análisis
    duracion_min = 1.0
    filtrados = []
    for i, (t, ac) in enumerate(cambios):
        fin = cambios[i + 1][0] if i + 1 < len(cambios) else t + duracion_min
        if fin - t >= duracion_min:
            if not filtrados or filtrados[-1][1] != ac:
                filtrados.append((t, ac))
    print(f"   {len(filtrados)} cambios de acorde detectados.")
    return filtrados


# ------------------------------------------------------------ alineación
def alinear(segmentos: list[dict], acordes: list[tuple[float, str]]) -> str:
    """Inserta cada acorde antes de la palabra cantada más cercana a su inicio."""
    lineas = []
    usados = set()
    sonando = None
    for seg in segmentos:
        partes = []
        for w in seg["palabras"]:
            # acordes que ocurren entre la palabra anterior y esta;
            # si hay varios, solo el último sigue sonando sobre esta palabra
            pendientes = [
                (j, ac) for j, (t, ac) in enumerate(acordes)
                if j not in usados and t <= w["t"] + 0.3
            ]
            for j, _ in pendientes:
                usados.add(j)
            if pendientes and pendientes[-1][1] != sonando:
                sonando = pendientes[-1][1]
                partes.append(f"[{sonando}]")
            partes.append(w["w"] + " ")
        lineas.append(("".join(partes)).strip())
    return "\n".join(lineas)


def _norm_palabra(w: str) -> str:
    w = unicodedata.normalize("NFKD", w.lower())
    w = "".join(c for c in w if not unicodedata.combining(c))
    return re.sub(r"[^a-zn0-9]", "", w)


def _alinear_dp(user: list[dict], ws: list[dict]) -> None:
    """Alinea las palabras de la letra del usuario con las palabras oídas por
    Whisper (Needleman-Wunsch con similitud difusa) y copia los tiempos."""
    import difflib

    A = [_norm_palabra(u["w"]) for u in user]
    B = [_norm_palabra(x["w"]) for x in ws]
    n, m = len(A), len(B)
    GAP = -0.4
    dp = np.zeros((n + 1, m + 1))
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)
    dp[1:, 0] = np.arange(1, n + 1) * GAP; bt[1:, 0] = 1
    dp[0, 1:] = np.arange(1, m + 1) * GAP; bt[0, 1:] = 2
    simcache: dict[tuple[str, str], float] = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            par = (A[i - 1], B[j - 1])
            s = simcache.get(par)
            if s is None:
                s = difflib.SequenceMatcher(None, *par).ratio()
                simcache[par] = s
            match = dp[i - 1][j - 1] + (s if s > 0.6 else -0.5)
            arriba = dp[i - 1][j] + GAP
            izq = dp[i][j - 1] + GAP
            mejor = max(match, arriba, izq)
            dp[i][j] = mejor
            bt[i][j] = 0 if mejor == match else (1 if mejor == arriba else 2)
    i, j = n, m
    while i > 0 and j > 0:
        if bt[i][j] == 0:
            if simcache.get((A[i - 1], B[j - 1]), 0) > 0.6:
                user[i - 1]["t"] = ws[j - 1]["t"]
            i, j = i - 1, j - 1
        elif bt[i][j] == 1:
            i -= 1
        else:
            j -= 1


def _interpolar_tiempos(user: list[dict]) -> None:
    """Estima el tiempo de las palabras sin ancla interpolando entre vecinas."""
    anclas = [(k, u["t"]) for k, u in enumerate(user) if u["t"] is not None]
    if not anclas:
        return
    for k, u in enumerate(user):
        if u["t"] is not None:
            continue
        prev = next((a for a in reversed(anclas) if a[0] < k), None)
        sig = next((a for a in anclas if a[0] > k), None)
        if prev and sig:
            frac = (k - prev[0]) / (sig[0] - prev[0])
            u["t"] = prev[1] + frac * (sig[1] - prev[1])
        elif prev:
            u["t"] = prev[1]
        else:
            u["t"] = sig[1]


def alinear_con_letra(letra: str, segmentos: list[dict],
                      acordes: list[tuple[float, str]]) -> str:
    """Coloca los acordes detectados sobre la letra dada por el usuario.

    Conserva las líneas tal cual (incluidos #Secciones y líneas vacías);
    solo inserta [Acorde] antes de la palabra donde cae cada cambio.
    """
    ws = [w for seg in segmentos for w in seg["palabras"]]
    lineas = letra.replace("\r", "").split("\n")
    user = []
    for i, ln in enumerate(lineas):
        t = ln.strip()
        if not t or t.startswith("#"):
            continue
        for tok in t.split():
            user.append({"linea": i, "w": tok, "t": None})
    if ws and user:
        print(">> Alineando tu letra con el audio...")
        _alinear_dp(user, ws)
        anclados = sum(1 for u in user if u["t"] is not None)
        print(f"   {anclados}/{len(user)} palabras ancladas al audio.")
        _interpolar_tiempos(user)

    por_linea: dict[int, list[dict]] = {}
    for u in user:
        por_linea.setdefault(u["linea"], []).append(u)

    out, usados, sonando = [], set(), None
    for i, ln in enumerate(lineas):
        t = ln.strip()
        if not t or t.startswith("#"):
            out.append(ln)
            continue
        partes = []
        for u in por_linea.get(i, []):
            if u["t"] is not None:
                pend = [
                    (j, ac) for j, (tt, ac) in enumerate(acordes)
                    if j not in usados and tt <= u["t"] + 0.3
                ]
                for j, _ in pend:
                    usados.add(j)
                if pend and pend[-1][1] != sonando:
                    sonando = pend[-1][1]
                    partes.append(f"[{sonando}]")
            partes.append(u["w"] + " ")
        out.append("".join(partes).strip())
    return "\n".join(out)


def limpiar_nombre(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^\w\- ]", "", texto).strip()[:60] or "alabanza"


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description="Transcriptor AcordGod")
    ap.add_argument("url", help="Link del video de YouTube")
    ap.add_argument("--titulo", default=None, help="Título de la alabanza")
    ap.add_argument("--letra", default=None,
                    help="Archivo .txt con la letra: los acordes se colocan sobre ella")
    args = ap.parse_args()

    wav, titulo_video = descargar_audio(args.url)
    titulo = args.titulo or titulo_video

    segmentos = transcribir_letra(wav)
    if not segmentos:
        print("!! No se detectó voz cantada. Se generarán solo los acordes.")
    acordes = detectar_acordes(wav)

    if args.letra:
        letra = Path(args.letra).read_text(encoding="utf-8")
        cuerpo = alinear_con_letra(letra, segmentos, acordes)
    elif segmentos:
        cuerpo = "#Transcripción automática\n" + alinear(segmentos, acordes)
    else:
        cuerpo = "#Acordes detectados\n" + " ".join(
            f"[{ac}]({int(t//60)}:{int(t%60):02d})" for t, ac in acordes
        )

    nombre = limpiar_nombre(titulo)
    txt = SALIDA / f"{nombre}.txt"
    txt.write_text(cuerpo, encoding="utf-8")

    cancion = {
        "id": int(__import__("time").time() * 1000),
        "title": titulo,
        "yt": args.url,
        "body": cuerpo,
        "transpose": 0,
        "marks": {},
    }
    js = SALIDA / f"{nombre}.acordgod.json"
    js.write_text(json.dumps([cancion], ensure_ascii=False, indent=2), encoding="utf-8")

    wav.unlink(missing_ok=True)
    print("\n== LISTO ==")
    print(f"   Letra con acordes : {txt}")
    print(f"   Archivo importable: {js}")
    print("   En la app usa el boton 'Importar' y elige el archivo .acordgod.json.")
    print("   Revisa y corrige el borrador: la deteccion automatica no es perfecta.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
