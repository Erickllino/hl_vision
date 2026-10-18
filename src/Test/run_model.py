#!/usr/bin/env python3
"""Mostra as predições do YOLO ao vivo na câmera (tecla q ou Esc para sair).

Exemplo: uv run --no-sync python src/Test/run_model.py --camera 0
Para o ONNX padrão, instale antes: uv pip install onnxruntime
"""

import argparse
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "Models" / "best_seg_orin_10.3.engine"


def camera_source(value: str) -> int | str:
    """Aceita índices de câmera (0, 1, ...) ou caminhos como /dev/video0."""
    if value.isdecimal():
        return int(value)
    return value


def confidence(value: str) -> float:
    try:
        conf = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("confiança deve ser um número entre 0 e 1") from exc
    if not 0 <= conf <= 1:
        raise argparse.ArgumentTypeError("confiança deve estar entre 0 e 1")
    return conf


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=camera_source, default=0,
                    help="Índice da câmera ou caminho do dispositivo (padrão: 0)")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                    help="Pesos YOLO (.onnx, .pt ou .engine); padrão: %(default)s")
    ap.add_argument("--task", choices=("detect", "segment", "classify", "pose", "obb", "semantic"),
                    default="detect", help="Tarefa do modelo (padrão: detect)")
    ap.add_argument("--conf", type=confidence, default=0.25,
                    help="Confiança mínima das predições (padrão: 0.25)")
    ap.add_argument("--imgsz", type=int, default=320,
                    help="Resolução de entrada; ONNX padrão tem tamanho fixo 320 (padrão: 320)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"[camera] Pesos não encontrados: {args.weights}")
    if args.imgsz <= 0:
        raise SystemExit("[camera] --imgsz deve ser positivo")
    if args.weights.suffix.lower() == ".onnx" and importlib.util.find_spec("onnxruntime") is None:
        raise SystemExit("[camera] ONNX requer onnxruntime. Instale com: uv pip install onnxruntime")

    import cv2
    from ultralytics import YOLO

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        raise SystemExit(f"[camera] Não foi possível abrir a câmera: {args.camera}")

    try:
        model = YOLO(str(args.weights), task=args.task)
        print(f"[camera] Pesos: {args.weights} | câmera: {args.camera} | sair: q ou Esc")
        while True:
            ok, frame = capture.read()
            if not ok:
                print("[camera] Falha ao ler a câmera; encerrando.")
                break

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            try:
                cv2.imshow("HL Vision - predições", result.plot())
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            except cv2.error as exc:
                raise SystemExit(
                    "[camera] Não foi possível exibir a janela; é necessário OpenCV com GUI "
                    "e uma sessão gráfica ativa."
                ) from exc
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # OpenCV sem GUI não oferece esta função.


if __name__ == "__main__":
    main()
