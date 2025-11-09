from __future__ import annotations
import os
import threading
import cv2 #type: ignore
import pyttsx3
from tkinter import Tk, Toplevel, Frame, Label, Button, filedialog, messagebox, BOTH, LEFT, TOP
from tkinter import ttk
from PIL import ImageTk #type: ignore
from algo import FaceEngine, match_embedding, draw_boxes_on_pil, DEFAULT_THRESHOLD

FIXED_THRESHOLD = DEFAULT_THRESHOLD
PHASES = ["Front profile", "Left profile", "Right profile"]
FRAMES_PER_PHASE = 5
FIXED_ENROLL_FRAMES = FRAMES_PER_PHASE * len(PHASES)


class FaceRecogGUI:
    def __init__(self, root: Tk, repo, engine: FaceEngine):
        self.root = root
        self.repo = repo
        self.engine = engine
        self.threshold = FIXED_THRESHOLD

        self.tts = pyttsx3.init()
        try:
            self.tts.setProperty('rate', 170)
        except Exception:
            pass

        root.title('Face Recognition — facenet-pytorch + MongoDB')
        top = Frame(root)
        top.pack(side=TOP, fill=BOTH, pady=6)

        Button(top, text='Enroll (Webcam)', command=self.enroll_via_webcam).pack(side=LEFT, padx=6)
        Button(top, text='Enroll (Images)', command=self.enroll_from_files).pack(side=LEFT, padx=6)
        Button(top, text='Predict Image', command=self.predict_image_dialog).pack(side=LEFT, padx=6)
        Button(top, text='Live Predict', command=self.live_predict_webcam).pack(side=LEFT, padx=6)
        Button(top, text='View History', command=self.view_history).pack(side=LEFT, padx=6)
        Button(top, text='Export History', command=self.export_history).pack(side=LEFT, padx=6)

        self.image_panel = Label(root)
        self.image_panel.pack(side=TOP, pady=8)
        self.result_label = Label(root, text='No prediction yet')
        self.result_label.pack(side=TOP)

    def enroll_from_files(self):
        files = filedialog.askopenfilenames(
            title='Select images',
            filetypes=[('Images','*.jpg *.jpeg *.png *.bmp *.webp')]
        )
        if not files:
            return
        name = simple_input_dialog(self.root, 'Enter person name')
        if not name:
            return

        def job():
            pid = self.repo.insert_person(name)
            count = 0
            for path in files:
                try:
                    pil = self.engine.image_to_pil(path)
                    embs, _ = self.engine.embed_faces(pil)
                except Exception:
                    continue
                for emb in embs:
                    self.repo.insert_embedding(pid, emb, path)
                    count += 1
            self.root.after(0, lambda: messagebox.showinfo('Enrollment', f'Added {count} embeddings for {name}'))

        threading.Thread(target=job, daemon=True).start()

    def enroll_via_webcam(self):
        name = simple_input_dialog(self.root, 'Enter person name')
        if not name:
            return

        def speak_blocking(text: str):
            try:
                eng = pyttsx3.init()
                try:
                    eng.setProperty('rate', 170)
                except Exception:
                    pass
                eng.say(text)
                eng.runAndWait()
                try:
                    eng.stop()
                except Exception:
                    pass
            except Exception as e:
                print('TTS error:', e)

        def job():
            pid = self.repo.insert_person(name)
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror('Webcam','Cannot open camera'))
                return

            total_collected = 0
            skip, idx = 3, 0

            for phase in PHASES:
                speak_blocking(f"Please show your {phase}")

                phase_count = 0
                while phase_count < FRAMES_PER_PHASE:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    idx += 1
                    if idx % skip != 0:
                        cv2.putText(frame, f'{phase}: {phase_count}/{FRAMES_PER_PHASE}  Total: {total_collected}/{FIXED_ENROLL_FRAMES}', (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        cv2.imshow('Enroll — press Q to stop', frame)
                        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                            cap.release(); cv2.destroyAllWindows()
                            self.root.after(0, lambda: messagebox.showinfo('Enrollment', f'Captured {total_collected} embeddings for {name}'))
                            return
                        continue

                    pil = self.engine.image_to_pil(frame)
                    embs, _ = self.engine.embed_faces(pil)
                    if embs:
                        self.repo.insert_embedding(pid, embs[0], f'webcam_{phase}')
                        phase_count += 1
                        total_collected += 1

                    cv2.putText(frame, f'{phase}: {phase_count}/{FRAMES_PER_PHASE}  Total: {total_collected}/{FIXED_ENROLL_FRAMES}', (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.imshow('Enroll — press Q to stop', frame)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        cap.release(); cv2.destroyAllWindows()
                        self.root.after(0, lambda: messagebox.showinfo('Enrollment', f'Captured {total_collected} embeddings for {name}'))
                        return

            cap.release()
            cv2.destroyAllWindows()
            self.root.after(0, lambda: messagebox.showinfo('Enrollment', f'Captured {total_collected} embeddings for {name}'))

        threading.Thread(target=job, daemon=True).start()

    def predict_image_dialog(self):
        path = filedialog.askopenfilename(title='Select image', filetypes=[('Images','*.jpg *.jpeg *.png *.bmp *.webp')])
        if not path:
            return

        def job():
            pil = self.engine.image_to_pil(path)
            embs, boxes = self.engine.embed_faces(pil)
            known = self.repo.fetch_all_embeddings()
            results = []
            for emb in embs:
                label, dist = match_embedding(emb, known, threshold=self.threshold)
                results.append((label, float(dist)))
                self.repo.insert_prediction(path, label, float(dist))

            img_disp = draw_boxes_on_pil(pil, boxes, results if results else None)
            img_disp.thumbnail((640,480))
            photo = ImageTk.PhotoImage(img_disp)

            def update():
                self.image_panel.config(image=photo)
                self.image_panel.image = photo
                self.result_label.config(text=''.join([f"{l} (dist={d:.3f})" for l,d in results]) if results else 'No faces detected')

            self.root.after(0, update)

        threading.Thread(target=job, daemon=True).start()

    def live_predict_webcam(self):
        def job():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror('Webcam', 'Cannot open camera'))
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                pil = self.engine.image_to_pil(frame)
                embs, boxes = self.engine.embed_faces(pil)
                known = self.repo.fetch_all_embeddings()
                labels = []
                for emb in embs:
                    label, dist = match_embedding(emb, known, threshold=self.threshold)
                    labels.append((label, float(dist)))

                if boxes is not None:
                    for i, b in enumerate(boxes):
                        x1, y1, x2, y2 = [int(x) for x in b]
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                        if i < len(labels):
                            cv2.putText(frame, f"{labels[i][0]} ({labels[i][1]:.2f})", (x1, max(0,y1-10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cv2.imshow('Live Predict — press Q to quit', frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            cap.release()
            cv2.destroyAllWindows()

        threading.Thread(target=job, daemon=True).start()

    def view_history(self):
        df = self.repo.fetch_predictions(limit=200)
        if df is None or df.empty:
            messagebox.showinfo('History', 'No predictions yet')
            return
        win = Toplevel(self.root)
        win.title('Prediction History')
        tv = ttk.Treeview(win, columns=list(df.columns), show='headings')
        for c in df.columns:
            tv.heading(c, text=c); tv.column(c, width=120)
        for _, row in df.iterrows():
            tv.insert('', 'end', values=list(row))
        tv.pack(fill=BOTH, expand=True)

    def export_history(self):
        df = self.repo.fetch_predictions(limit=10000)
        if df is None or df.empty:
            messagebox.showinfo('Export', 'No predictions to export')
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not path:
            return
        df.to_csv(path, index=False)
        messagebox.showinfo('Export', f'Exported {len(df)} rows to {path}')


def simple_input_dialog(parent, prompt):
    res = {'v': None}
    dlg = Toplevel(parent); dlg.title(prompt)
    Label(dlg, text=prompt).pack(pady=6)
    from tkinter import Entry, StringVar
    var = StringVar(); Entry(dlg, textvariable=var).pack(pady=6)
    Button(dlg, text='OK', command=lambda: (res.update({'v': var.get().strip()}), dlg.destroy())).pack(pady=6)
    parent.wait_window(dlg)
    return res['v']
