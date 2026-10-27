"""ROS 2 bag -> unmodified vision_node -> X11 diagnostic UI."""

import argparse
from collections import OrderedDict
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
import yaml
from PIL import Image, ImageTk
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.srv import SetParametersAtomically
from sensor_msgs.msg import Image as RosImage, CompressedImage
from vision_interface.msg import Detections, LineSegments

from replay_config import FIELDS, load, write_config


def stamp(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def decode(msg):
    if isinstance(msg, CompressedImage):
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    if msg.encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"encoding não suportado: {msg.encoding}")
    array = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    image = array[:, : msg.width * 3].reshape(msg.height, msg.width, 3).copy()
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else image


class Monitor:
    def __init__(self, topic):
        self.node = rclpy.create_node("hl_vision_replay_monitor")
        self.lock = threading.Lock()
        self.frames = OrderedDict()
        self.overlays = OrderedDict()
        self.processed = None
        self.pre_count = 0
        self.mask_count = 0
        self.errors = None
        image_type = CompressedImage if "compressed" in topic else RosImage
        self.subs = [
            self.node.create_subscription(image_type, topic, self.raw, qos_profile_sensor_data),
            self.node.create_subscription(RosImage, "/booster_camera/preprocess_video", self.pre, qos_profile_sensor_data),
            self.node.create_subscription(CompressedImage, "/booster_camera/segmentation_overlay/compressed", self.mask, qos_profile_sensor_data),
            self.node.create_subscription(Detections, "/booster_soccer/detection", self.det, 10),
            self.node.create_subscription(LineSegments, "/booster_soccer/line_segments", self.seg, 10),
        ]
        self.executor = SingleThreadedExecutor()
        self.parameter_client = self.node.create_client(SetParametersAtomically, "/vision_node/set_parameters_atomically")
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def raw(self, msg):
        try:
            image = decode(msg)
            if image is None:
                raise ValueError("imagem JPEG inválida")
            with self.lock:
                self.frames[stamp(msg.header)] = [image, None, None]
                while len(self.frames) > 60:
                    self.frames.popitem(last=False)
        except Exception as exc:
            self.errors = str(exc)

    def pre(self, msg):
        try:
            image = decode(msg)
            with self.lock:
                self.processed = (stamp(msg.header), image)
                self.pre_count += 1
        except Exception as exc:
            self.errors = str(exc)

    def mask(self, msg):
        try:
            image = decode(msg)
            with self.lock:
                self.overlays[stamp(msg.header)] = image
                self.mask_count += 1
                while len(self.overlays) > 60:
                    self.overlays.popitem(last=False)
        except Exception as exc:
            self.errors = str(exc)

    def det(self, msg):
        with self.lock:
            if stamp(msg.header) in self.frames:
                self.frames[stamp(msg.header)][1] = list(msg.detected_objects)

    def seg(self, msg):
        with self.lock:
            if stamp(msg.header) in self.frames:
                self.frames[stamp(msg.header)][2] = list(msg.coordinates_uv)

    def snapshot(self):
        with self.lock:
            if not self.frames:
                return None, None, None, None
            newest = next(reversed(self.frames))
            # Keep a rare segmentation overlay visible briefly even when the
            # detector publishes newer frames sooner than the segmentor.
            overlay_key = next((k for k in reversed(self.overlays) if
                                k in self.frames and 0 <= newest - k <= 1_000_000_000), None)
            key = overlay_key or next((k for k in reversed(self.frames) if
                                       self.frames[k][1] is not None or self.frames[k][2] is not None), newest)
            image, detections, segments = self.frames[key]
            pre = self.processed[1].copy() if self.processed and self.processed[0] == key else None
            overlay = self.overlays.get(key)
            return (overlay if overlay is not None else image).copy(), detections, segments, pre

    def close(self):
        self.executor.shutdown()
        self.thread.join(timeout=3)
        self.node.destroy_node()


def stop(proc):
    if proc is not None and hasattr(proc, "stop"):
        proc.stop()
        return
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


class App:
    def __init__(self, args, base, values):
        self.args, self.base = args, base
        self.last_applied_values = values.copy()
        self.root = tk.Tk()
        self.root.title("hl_vision | rosbag + hsl-player")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        signal.signal(signal.SIGTERM, lambda *_: self.root.after(0, self.close))
        self.vision = self.bag = None
        self.closing = False
        self.applying = False
        self.waiting_for_pre = None
        self.parameter_future = None
        self.notifications = queue.SimpleQueue()
        self.input_topics = [topic for topic in (args.color_topic, args.intrin_topic, args.pose_topic) if topic]
        self.monitor = Monitor(args.color_topic)
        self.vars = {}
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        for key, (_, _, kind) in FIELDS.items():
            var = tk.BooleanVar(value=values[key]) if kind is bool else tk.StringVar(value=str(values[key]))
            self.vars[key] = var
            if kind is bool:
                tk.Checkbutton(frame, text=key, variable=var).pack(anchor="w")
            else:
                row = tk.Frame(frame)
                row.pack(anchor="w")
                tk.Label(row, text=key, width=21, anchor="w").pack(side=tk.LEFT)
                tk.Entry(row, textvariable=var, width=9).pack(side=tk.LEFT)
        self.apply_button = tk.Button(frame, text="Aplicar filtros", command=self.apply)
        self.apply_button.pack(pady=12)
        self.status = tk.StringVar(value="Iniciando...")
        tk.Label(frame, textvariable=self.status, wraplength=250, justify=tk.LEFT).pack()
        self.progress = ttk.Progressbar(frame, orient="horizontal", length=245, mode="determinate", maximum=100)
        self.progress.pack(pady=(14, 2))
        self.progress_text = tk.StringVar(value="Aguardando duração da bag...")
        tk.Label(frame, textvariable=self.progress_text).pack()
        self.masks_text = tk.StringVar(value="Máscaras previstas: 0 frames")
        tk.Label(frame, textvariable=self.masks_text).pack()
        panes = tk.Frame(self.root)
        panes.pack(side=tk.LEFT)
        self.raw_label = tk.Label(panes, text="Aguardando RAW")
        self.raw_label.pack(side=tk.LEFT)
        self.pre_label = tk.Label(panes, text="Aguardando pré-processada")
        self.pre_label.pack(side=tk.LEFT)
        self.apply(initial=True)
        try:
            if args.bag.is_file() and args.bag.suffix == ".mcap":
                from mcap_replay import McapReplay
                self.bag = McapReplay(args.bag, self.input_topics,
                                      args.rate, args.logs / "bag.log")
            elif args.bag.is_dir() and list(args.bag.glob("*.mcap")) and "mcap" not in subprocess.check_output(
                    ["ros2", "bag", "list", "storage"], text=True).split():
                from mcap_replay import McapReplay
                bags = list(args.bag.glob("*.mcap"))
                if len(bags) != 1:
                    raise ValueError("Replay Python aceita uma bag com um arquivo MCAP; instale rosbag2_storage_mcap para bags divididas")
                self.bag = McapReplay(bags[0], self.input_topics,
                                      args.rate, args.logs / "bag.log")
            else:
                # Replay only sensor inputs, never recorded detections or robot control.
                self.bag = self.start(["ros2", "bag", "play", str(args.bag), "--loop", "--rate", str(args.rate),
                                       "--topics", *self.input_topics], "bag")
        except OSError:
            self.close()
            raise
        self.root.after(100, self.refresh)

    def start(self, cmd, log_name, cwd=None):
        log = open(self.args.logs / f"{log_name}.log", "ab", buffering=0)
        try:
            return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=cwd)
        finally:
            log.close()

    def apply(self, initial=False):
        if self.applying:
            return
        try:
            values = {key: var.get() if kind is bool else kind(var.get()) for key, var in self.vars.items()
                      for kind in (FIELDS[key][2],)}
            write_config(self.base, values, self.args.logs / "vision_replay.yaml")
        except (ValueError, OSError) as exc:
            messagebox.showerror("Configuração", str(exc))
            return
        self.applying = True
        self.apply_button.config(state=tk.DISABLED)
        self.pending_values = values
        self.monitor.errors = None
        if initial:
            self.status.set("Iniciando visão e carregando modelos...")
            self.restart()
        elif self.monitor.parameter_client.service_is_ready() and self.vision and self.vision.poll() is None:
            config = yaml.safe_load((self.args.logs / "vision_replay.yaml").read_text())
            value = yaml.safe_dump({"line_preprocessing": config["segmentation_model"]["line_preprocessing"]})
            request = SetParametersAtomically.Request()
            request.parameters = [Parameter("segmentation_preprocessing_yaml", value=value).to_parameter_msg()]
            self.parameter_future = self.monitor.parameter_client.call_async(request)
            self.status.set("Enviando novos filtros ao hsl-player (sem reiniciar)...")
        else:
            self.restart_async()

    def restart_async(self):
        self.status.set("Aplicando: reiniciando visão e carregando modelos...")
        self.restart_thread = threading.Thread(target=self.restart, daemon=True)
        self.restart_thread.start()

    def restart(self):
        try:
            previous = self.vision
            self.vision = None
            self.waiting_for_pre = None
            stop(previous)
            with self.monitor.lock:
                self.monitor.frames.clear()
                self.monitor.overlays.clear()
                self.monitor.processed = None
                pre_count = self.monitor.pre_count
            if self.closing:
                return
            # vision_node reads argv[1] as template and argv[2] as overlay; the
            # full generated template avoids editing installed hsl-player files.
            cmd = [str(self.args.vision_bin), str(self.args.logs / "vision_replay.yaml"),
                   # This hsl-player version dereferences its logger even when
                   # save_data=false. Keep it enabled at the minimum rate.
                   "--ros-args", "-p", "save_data:=true", "-p", "save_fps:=1", "-p", "save_depth:=false",
                   "-p", f"offline_mode:={'true' if self.args.pose_topic == '/booster_soccer/t_head2base' else 'false'}",
                   "-p", f"color_topic:={self.args.color_topic}"]
            if self.args.intrin_topic:
                cmd += ["-p", f"intrin_topic:={self.args.intrin_topic}"]
            if self.args.det_model:
                cmd += ["-p", f"detection_model_path:={self.args.det_model}"]
            if self.args.seg_model:
                cmd += ["-p", f"segmentation_model_path:={self.args.seg_model}"]
            self.vision = self.start(cmd, "vision", cwd=self.args.config.resolve().parents[3])
            self.waiting_for_pre = (pre_count, time.monotonic())
            self.notifications.put(("status", "Visão iniciada; aguardando frame pré-processado..."))
        except (ValueError, OSError) as exc:
            self.notifications.put(("error", str(exc)))

    def apply_failed(self, error):
        self.applying = False
        self.apply_button.config(state=tk.NORMAL)
        self.status.set(f"Falha ao aplicar: {error}")
        write_config(self.base, self.last_applied_values, self.args.logs / "vision_replay.yaml")

    def refresh(self):
        self.masks_text.set(f"Máscaras previstas: {self.monitor.mask_count} frames")
        if self.parameter_future and self.parameter_future.done():
            future, self.parameter_future = self.parameter_future, None
            try:
                result = future.result().result
                if result.successful:
                    with self.monitor.lock:
                        pre_count = self.monitor.pre_count
                    self.waiting_for_pre = (pre_count, time.monotonic())
                    self.status.set("Filtros atualizados sem reinício; aguardando frame de confirmação...")
                elif "declare" in result.reason.lower():
                    # Running an older binary: restart once into the new build.
                    self.restart_async()
                else:
                    self.apply_failed(result.reason)
            except Exception as exc:
                self.apply_failed(str(exc))
        while not self.notifications.empty():
            kind, value = self.notifications.get_nowait()
            if kind == "error":
                self.apply_failed(value)
            else:
                self.status.set(value)
        if hasattr(self.bag, "progress"):
            elapsed, duration, loops = self.bag.progress()
            if duration > 0:
                self.progress["value"] = min(100, elapsed * 100 / duration)
                self.progress_text.set(f"Bag: {int(elapsed // 60):02d}:{int(elapsed % 60):02d} / "
                                       f"{int(duration // 60):02d}:{int(duration % 60):02d}  •  volta {loops + 1}")
        elif self.bag:
            self.progress_text.set("Progresso: disponível para MCAP sem plugin ROS")
        raw, det, seg, pre = self.monitor.snapshot()
        if raw is not None:
            for obj in det or []:
                cv2.rectangle(raw, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), (0, 255, 255), 2)
                cv2.putText(raw, f"{obj.label} {obj.confidence:.0f}%", (obj.xmin, max(15, obj.ymin - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
            for i in range(0, len(seg or []) - 3, 4):
                cv2.line(raw, (int(seg[i]), int(seg[i + 1])), (int(seg[i + 2]), int(seg[i + 3])), (255, 0, 255), 2)
            self.draw(self.raw_label, raw, "RAW + máscaras + caixas + segmentos")
        if pre is not None:
            self.draw(self.pre_label, pre, "Entrada da segmentação")
        if self.monitor.errors:
            self.status.set(self.monitor.errors)
        elif self.bag and self.bag.poll() is not None:
            self.status.set(f"rosbag terminou ({self.bag.returncode}); veja bag.log")
        elif self.vision and self.vision.poll() is not None:
            self.status.set(f"vision_node terminou ({self.vision.returncode}); veja vision.log")
            if self.applying:
                self.applying = False
                self.apply_button.config(state=tk.NORMAL)
        elif self.waiting_for_pre:
            with self.monitor.lock:
                done = self.monitor.pre_count > self.waiting_for_pre[0]
            if done:
                self.waiting_for_pre = None
                self.applying = False
                self.last_applied_values = self.pending_values.copy()
                self.apply_button.config(state=tk.NORMAL)
                self.status.set("Aplicado ✓ — novo frame processado com os filtros selecionados")
            elif time.monotonic() - self.waiting_for_pre[1] > 10:
                self.status.set("Visão carregada; aguardando próximo frame da bag para confirmar...")
        self.root.after(100, self.refresh)

    def draw(self, label, bgr, title):
        h, w = bgr.shape[:2]
        scale = min(720 / w, 540 / h)
        bgr = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        label.configure(image=photo, text=title, compound=tk.TOP)
        label.image = photo

    def close(self):
        self.closing = True
        stop(self.bag)
        restart_thread = getattr(self, "restart_thread", None)
        if restart_thread is not None and restart_thread.is_alive():
            restart_thread.join(timeout=10)
        stop(self.vision)
        self.monitor.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Replay de rosbag no hsl-player (ROS 2 + X11)")
    parser.add_argument("bag", type=Path)
    parser.add_argument("--config", type=Path, required=True, help="vision.yaml do hsl-player")
    parser.add_argument("--vision-bin", type=Path, required=True, help="binário vision_node compilado")
    parser.add_argument("--color-topic", required=True, help="tópico de imagem presente na bag")
    parser.add_argument("--intrin-topic", help="CameraInfo presente na bag; sem ele, usa intrínsecos do YAML")
    parser.add_argument("--pose-topic", choices=("/head_pose", "/booster_soccer/t_head2base"),
                        help="pose presente na bag; sem ela, geometria 3D não é confiável")
    parser.add_argument("--det-model", type=Path, help="modelo de detecção (padrão: k1_realsense_10.3.engine)")
    parser.add_argument("--seg-model", type=Path)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--logs", type=Path, default=Path("runs/replay"))
    args = parser.parse_args()
    if not args.bag.exists() or not args.vision_bin.is_file() or not args.config.is_file():
        parser.error("bag, config e vision-bin devem existir")
    if args.rate <= 0 or not os.environ.get("DISPLAY"):
        parser.error("rate deve ser > 0 e DISPLAY deve estar disponível (ssh -X)")
    if not os.environ.get("ROS_DOMAIN_ID"):
        parser.error("defina ROS_DOMAIN_ID exclusivo para isolar a câmera real")
    args.bag = args.bag.resolve()
    args.vision_bin = args.vision_bin.resolve()
    args.det_model = (args.det_model or args.config.resolve().parents[3] /
                      "src/vision/model/k1_realsense_10.3.engine").resolve()
    if not args.det_model.is_file():
        parser.error(f"modelo de detecção não encontrado: {args.det_model}")
    args.logs = args.logs.resolve()
    args.logs.mkdir(parents=True, exist_ok=True)
    base, values = load(args.config)
    rclpy.init()
    try:
        app = App(args, base, values)
        app.root.mainloop()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
