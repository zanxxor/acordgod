# -*- coding: utf-8 -*-
"""
AcordGod - Servidor local del transcriptor.

Deja este servidor corriendo (doble clic en servidor.bat) y la app web
mostrará el botón "Transcribir" activo. La app le envía el link de YouTube
y este servidor devuelve la canción transcrita (letra + acordes).
"""
import cgi
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pdf_acordes
import transcribir as T

PUERTO = 8765
_lock = threading.Lock()  # una transcripción a la vez


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        # permite que la página https (GitHub Pages) hable con localhost
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def _json(self, code: int, data: dict):
        cuerpo = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/estado":
            self._json(200, {"ok": True, "ocupado": _lock.locked()})
        else:
            self._json(404, {"error": "ruta desconocida"})

    def do_POST(self):
        if self.path == "/extraer_pdf":
            self._extraer_pdf()
            return
        if self.path != "/transcribir":
            self._json(404, {"error": "ruta desconocida"})
            return
        try:
            largo = int(self.headers.get("Content-Length", 0))
            datos = json.loads(self.rfile.read(largo) or b"{}")
            url = (datos.get("url") or "").strip()
            if not url:
                self._json(400, {"error": "Falta el link de YouTube."})
                return
            if not _lock.acquire(blocking=False):
                self._json(409, {"error": "Ya hay una transcripción en curso. Espera a que termine."})
                return
            try:
                print(f"\n>> Transcribiendo: {url}")
                wav, titulo_video = T.descargar_audio(url)
                titulo = datos.get("titulo") or titulo_video
                letra = (datos.get("letra") or "").strip()
                permitidos = datos.get("acordes_pdf") or None
                segmentos = T.transcribir_letra(wav)
                acordes = T.detectar_acordes(wav, permitidos)
                if letra:
                    cuerpo = T.alinear_con_letra(letra, segmentos, acordes)
                elif segmentos:
                    cuerpo = "#Transcripción automática\n" + T.alinear(segmentos, acordes)
                else:
                    cuerpo = "#Acordes detectados\n" + " ".join(
                        f"[{ac}]({int(t // 60)}:{int(t % 60):02d})" for t, ac in acordes
                    )
                wav.unlink(missing_ok=True)
                print(">> Transcripción enviada a la app.")
                self._json(200, {"ok": True, "titulo": titulo, "cuerpo": cuerpo})
            finally:
                _lock.release()
        except Exception as e:  # noqa: BLE001 - se reporta el error a la app
            print(f"!! Error: {e}")
            try:
                self._json(500, {"error": str(e)})
            except Exception:
                pass

    def _extraer_pdf(self):
        import os
        import tempfile

        try:
            ctype, pdict = cgi.parse_header(self.headers.get("Content-Type", ""))
            if ctype != "multipart/form-data":
                self._json(400, {"error": "Se esperaba un archivo PDF (multipart/form-data)."})
                return
            pdict["boundary"] = pdict["boundary"].encode("utf-8")
            pdict["CONTENT-LENGTH"] = self.headers.get("Content-Length")
            campos = cgi.parse_multipart(self.rfile, pdict)
            archivos = campos.get("file")
            if not archivos:
                self._json(400, {"error": "No se recibió ningún archivo."})
                return
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(archivos[0])
                ruta = tmp.name
            try:
                print("\n>> Extrayendo acordes del PDF...")
                cuerpo = pdf_acordes.extraer(ruta)
                if not cuerpo.strip() or pdf_acordes.es_solo_diagramas(cuerpo):
                    info = pdf_acordes.extraer_secuencia(ruta)
                    if not info["acordes"]:
                        self._json(200, {"ok": True, "cuerpo": "", "vacio": True})
                        return
                    print(f">> El PDF es un diagrama de acordes (sin letra). Acordes: {info['acordes']}")
                    self._json(200, {"ok": True, "modo": "acordes",
                                      "acordes": info["acordes"], "capo": info["capo"]})
                    return
            finally:
                os.unlink(ruta)
            print(">> Letra y acordes extraídos del PDF enviados a la app.")
            self._json(200, {"ok": True, "modo": "letra", "cuerpo": cuerpo})
        except Exception as e:  # noqa: BLE001
            print(f"!! Error: {e}")
            self._json(500, {"error": str(e)})

    def log_message(self, *args):  # silencia el log por petición
        pass


if __name__ == "__main__":
    print("=" * 56)
    print("  AcordGod - Servidor del transcriptor")
    print(f"  Escuchando en http://localhost:{PUERTO}")
    print("  Deja esta ventana abierta mientras usas la app.")
    print("  (Ctrl+C o cerrar la ventana para detener)")
    print("=" * 56)
    ThreadingHTTPServer(("127.0.0.1", PUERTO), Handler).serve_forever()
