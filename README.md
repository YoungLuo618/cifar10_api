Author: Jianyang Luo

1.Project Overview：This project implements and deploys a CNN trained on the CIFAR-10 dataset using PyTorch.
A FastAPI service is built around the trained model to provide an HTTP API endpoint for real-time image classification.
The project also includes a Dockerfile for containerized deployment to ensure the API can be easily built and run on any machine.

2.Project Structure
cifar10_api/
app/
 main.py             # FastAPI app: /, /health, /predict endpoints
models/
 cnn.py              # CNN model architecture (for CIFAR-10)
train.py               # Training script (saves model.pth)
infer_utils.py         # Image preprocessing & prediction utilities
requirements.txt       # Python dependencies
Dockerfile             # Container build instructions
model.pth              # Trained weights file

# Run Locally:
#Create and activate virtual environment
python -m venv venv
#Windows
venv\Scripts\activate
#macOS/Linux
source venv/bin/activate
#Install dependencies
pip install -r requirements.txt
#Train model (optional if model.pth exists)
python train.py
#Start FastAPI server
uvicorn app.main:app --reload
#Access API docs:
http://127.0.0.1:8000/docs

# Run with Docker 
#Build the Docker image
docker build -t cifar10-api .
#Run the container
docker run -d -p 8000:80 --name cifar10_api cifar10-api
#If you prefer to mount an external model file:
docker run -d -p 8000:80 \
  -v C:\path\to\model.pth:/app/model.pth \
  --name cifar10_api cifar10-api
#Open the API
Docs: http://127.0.0.1:8000/docs
Health check: http://127.0.0.1:8000/health
Root endpoint: http://127.0.0.1:8000/

