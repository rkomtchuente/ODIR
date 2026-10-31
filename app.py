from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import torch
import torchvision.transforms as T
from model import ODIRclassifier  # your model file

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ODIRclassifier(n_classes=8, attr_dim=2)
model.load_state_dict(torch.load("odir_model.pth", map_location=device))
model.eval().to(device)

tf = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

diagnostic = ['Normal','Diabetes','Glaucoma','Cataract','AMD','Hypertension','Myopia','Other']

@app.post("/diagnose/")
async def diagnose(left_eye: UploadFile = File(...),
                   right_eye: UploadFile = File(...),
                   age: float = Form(...),
                   gender: int = Form(...)):

    L = tf(Image.open(left_eye.file).convert("RGB")).unsqueeze(0).to(device)
    R = tf(Image.open(right_eye.file).convert("RGB")).unsqueeze(0).to(device)
    meta = torch.tensor([[age/100.0, gender]], dtype=torch.float32).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(L,R,meta))[0].cpu().tolist()

    return {diagnostic[i]: round(p,3) for i,p in enumerate(probs)}
