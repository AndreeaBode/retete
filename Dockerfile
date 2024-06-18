# Utilizează o imagine de bază oficială Python
FROM python:3.10-slim

# Setează directorul de lucru
WORKDIR /app

# Copiază fișierele necesare
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY mlb.pkl mlb.pkl

# Instalează dependențele
RUN pip install --no-cache-dir -r requirements.txt

# Expune portul pe care rulează aplicația Flask
EXPOSE 5000

# Comanda pentru a rula aplicația
CMD ["python", "app.py"]
