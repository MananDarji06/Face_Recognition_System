import numpy as np # type: ignore
import cv2 # type: ignore
from PIL import Image # type: ignore
import torch # type: ignore
from facenet_pytorch import MTCNN, InceptionResnetV1 # type: ignore

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_THRESHOLD = 0.6

class FaceEngine:
    def __init__(self, device: str | None = None):
        self.device = device or DEVICE
        self.detector = MTCNN(keep_all=True, device=self.device)
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    @staticmethod
    def image_to_pil(img_or_path):
        if isinstance(img_or_path, str):
            return Image.open(img_or_path).convert('RGB')
        rgb = cv2.cvtColor(img_or_path, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def embed_faces(self, pil_img: Image.Image):
        faces = self.detector(pil_img)
        if faces is None:
            return [], None
        with torch.no_grad():
            embs = self.embedder(faces.to(self.device)).cpu().numpy()
        boxes, _ = self.detector.detect(pil_img)
        return [e for e in embs], boxes


def _cosine_distance(query_vec: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    G = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-10)
    return 1.0 - (G @ q)


def match_embedding(query_emb: np.ndarray, known: list[tuple[str, np.ndarray]], threshold: float = DEFAULT_THRESHOLD):
    if not known:
        return 'Unknown', None
    names = [n for n, _ in known]
    gallery = np.stack([e for _, e in known], axis=0)
    dists = _cosine_distance(query_emb, gallery)
    i = int(np.argmin(dists))
    best_name, best_dist = names[i], float(dists[i])
    return (best_name if best_dist <= threshold else 'Unknown', best_dist)


def draw_boxes_on_pil(pil_img: Image.Image, boxes, labels=None):
    if boxes is None:
        return pil_img
    arr = np.array(pil_img.copy())
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(x) for x in b]
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if labels and i < len(labels):
            text = f"{labels[i][0]} ({labels[i][1]:.2f})"
            cv2.putText(arr, text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return Image.fromarray(arr)