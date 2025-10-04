# ใช้ Python image เบา ๆ
FROM python:3.10-slim

# ตั้ง working directory
WORKDIR /app

# copy requirements.txt ก่อน
COPY requirements.txt .

# ติดตั้ง dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy code ทั้งหมด
COPY . .

# expose port (Hugging Face ใช้ 7860 เป็น default)
ENV PORT=7860

# run flask app
CMD ["python", "app.py"]
