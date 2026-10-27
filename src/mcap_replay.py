"""ROS 2 MCAP playback for robots without rosbag2_storage_mcap installed.

Only the three explicitly requested sensor topics are deserialized/published.
The node uses the same ROS_DOMAIN_ID as the diagnostic interface.
"""

import threading
import time

from mcap.reader import make_reader
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


class McapReplay:
    def __init__(self, path, topics, rate, log_path):
        self.path, self.topics, self.rate, self.log_path = path, topics, rate, log_path
        self.node = rclpy.create_node("hl_vision_mcap_replay")
        self.stopping = threading.Event()
        self.returncode = None
        self.publishers = {}
        self.lock = threading.Lock()
        self.elapsed = 0.0
        self.duration = 0.0
        self.loops = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def poll(self):
        return self.returncode

    def progress(self):
        with self.lock:
            return self.elapsed, self.duration, self.loops

    def run(self):
        try:
            while not self.stopping.is_set():
                count = 0
                with open(self.path, "rb") as f:
                    reader = make_reader(f)
                    summary = reader.get_summary()
                    if summary and summary.statistics:
                        with self.lock:
                            self.duration = (summary.statistics.message_end_time - summary.statistics.message_start_time) / 1e9
                    start_ns = None
                    start_wall = None
                    for schema, channel, message in reader.iter_messages(topics=self.topics, log_time_order=True):
                        if self.stopping.is_set():
                            break
                        if start_ns is None:
                            start_ns, start_wall = message.log_time, time.monotonic()
                            with self.lock:
                                self.elapsed = 0.0
                        due = start_wall + (message.log_time - start_ns) / (1e9 * self.rate)
                        if self.stopping.wait(max(0, due - time.monotonic())):
                            break
                        with self.lock:
                            self.elapsed = (message.log_time - start_ns) / 1e9
                        if channel.topic not in self.publishers:
                            msg_type = get_message(schema.name)
                            self.publishers[channel.topic] = (self.node.create_publisher(msg_type, channel.topic, 10), msg_type)
                        pub, msg_type = self.publishers[channel.topic]
                        pub.publish(deserialize_message(message.data, msg_type))
                        count += 1
                if count == 0 and not self.stopping.is_set():
                    raise RuntimeError("MCAP não contém os tópicos de imagem, CameraInfo ou pose selecionados")
                with self.lock:
                    self.loops += 1
        except Exception as exc:
            with open(self.log_path, "a") as f:
                f.write(f"Erro no replay MCAP: {exc!r}\n")
            self.returncode = 1
        else:
            self.returncode = 0

    def stop(self):
        self.stopping.set()
        self.thread.join(timeout=10)
        self.node.destroy_node()
