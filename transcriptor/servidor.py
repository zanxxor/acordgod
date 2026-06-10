# -*- coding: utf-8 -*-
"""
AcordGod - Servidor local del transcriptor.

Deja este servidor corriendo (doble clic en servidor.bat) y la app web
mostrará el botón "Transcribir" activo. La app le envía el link de YouTube
y este servidor devuelve la canción transcrita (letra + acordes).
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
                segmentos = T.transcribir_letra(wav)
                acordes = T.detectar_acordes(wav)
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
