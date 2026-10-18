"""Dataset-level metrics in image coordinates (not field geometry)."""
import numpy as np


def divide(a, b):
    return float(a / b) if b else None


def box_iou(box, others):
    others = np.asarray(others, dtype=float).reshape(-1, 4)
    lo = np.maximum(box[:2], others[:, :2])
    hi = np.minimum(box[2:], others[:, 2:])
    intersection = np.maximum(hi - lo, 0).prod(axis=1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    union = area + (others[:, 2:] - others[:, :2]).prod(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def detection_metrics(truth, predictions, classes):
    """GT rows: [class,x1,y1,x2,y2], prediction rows: same + score.

    Score-ordered class-aware matching and 101-point interpolated AP.
    No COCO crowd/area/maxDets semantics; all supplied predictions evaluated.
    """
    report = []
    for cls, name in enumerate(classes):
        gt = {key: [row[1:5] for row in rows if row[0] == cls]
              for key, rows in truth.items()}
        total = sum(map(len, gt.values()))
        ranked = sorted(((key, row) for key, rows in predictions.items()
                         for row in rows if row[0] == cls), key=lambda item: -item[1][5])
        aps = []
        operating = None
        for threshold in np.linspace(.5, .95, 10):
            used = {key: set() for key in gt}
            hits = []
            for key, row in ranked:
                overlaps = box_iou(np.asarray(row[1:5]), gt[key])
                candidates = [i for i in range(len(overlaps)) if i not in used[key]]
                best = max(candidates, key=lambda i: overlaps[i]) if candidates else None
                hit = best is not None and overlaps[best] >= threshold
                if hit:
                    used[key].add(best)
                hits.append(int(hit))
            tp = np.cumsum(hits)
            fp = np.arange(1, len(hits) + 1) - tp
            precision = tp / np.maximum(tp + fp, 1)
            recall = tp / max(total, 1)
            aps.append(float(np.mean([max(precision[recall >= r], default=0.)
                                      for r in np.linspace(0, 1, 101)])) if total else None)
            if operating is None:
                n_tp = int(sum(hits))
                operating = {"tp": n_tp, "fp": len(hits) - n_tp, "fn": total - n_tp,
                             "precision": divide(n_tp, len(hits)), "recall": divide(n_tp, total)}
        report.append({"class": name, "ground_truth": total, "ap50": aps[0],
                       "ap50_95": float(np.mean(aps)) if total else None,
                       "at_iou50": operating})
    valid = [r for r in report if r["ground_truth"]]
    return {"per_class": report,
            "map50": float(np.mean([r["ap50"] for r in valid])) if valid else None,
            "map50_95": float(np.mean([r["ap50_95"] for r in valid])) if valid else None}


class MaskMetrics:
    """Per-class binary unions of instances; ignored pixels excluded."""
    def __init__(self, classes):
        self.classes = classes
        self.counts = np.zeros((len(classes), 3), dtype=np.int64)

    def add(self, truth, predictions):
        valid = truth != 255
        for cls in range(len(self.classes)):
            target = truth == cls + 1
            pred = predictions[cls] & valid
            self.counts[cls] += [np.count_nonzero(pred & target),
                                 np.count_nonzero(pred & ~target),
                                 np.count_nonzero(target & ~pred)]

    def result(self):
        rows = []
        for name, (tp, fp, fn) in zip(self.classes, self.counts):
            rows.append({"class": name, "tp_pixels": int(tp), "fp_pixels": int(fp),
                         "fn_pixels": int(fn), "iou": divide(tp, tp + fp + fn),
                         "dice": divide(2 * tp, 2 * tp + fp + fn),
                         "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn)})
        valid = [r["iou"] for r in rows if r["iou"] is not None]
        return {"per_class": rows, "miou": float(np.mean(valid)) if valid else None}


def latency(values):
    return {"samples": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99))}
