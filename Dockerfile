# Imagen base: Python 3.12.5 slim (liviana)
FROM python:3.12.5-slim

# Establecer directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar requirements.txt primero (optimización de cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY app/ ./app/

# Copiar los modelos entrenados
COPY models/ ./models/

# Exponer el puerto 8000
EXPOSE 8000

# Comando para iniciar la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]