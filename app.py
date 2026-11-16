import gradio as gr
from ultralytics import YOLO

model = YOLO('kvasir_best.pt')


def detect_polyps(image):
    results = model.predict(image, conf=0.5)
    annotated = results[0].plot()
    return annotated[:, :, ::-1]  # BGR to RGB for Gradio


demo = gr.Interface(
    fn=detect_polyps,
    inputs=gr.Image(type='numpy', label='Endoscopy frame'),
    outputs=gr.Image(type='numpy', label='Detected polyps'),
    title='Polyp Detection — YOLO11 on Kvasir-SEG',
    description='Upload a GI endoscopy frame to detect polyps. Not for clinical use — research demo only.',
    examples=None
)

if __name__ == '__main__':
    demo.launch()
