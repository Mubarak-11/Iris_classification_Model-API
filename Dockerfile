FROM python:3.10-slim

#the directory inside the image
WORKDIR /app

#copy the requriements
COPY requirements.txt requirements.txt

#build the requirements
RUN pip install --no-cache-dir -r requirements.txt

#fast api connections: app.py
#package + helpers used in server directory
COPY server/ server/ 

#model + its weights
COPY iris_proj.py ./
COPY iris_model_weights.pkl .

#port/connection
EXPOSE 8000
CMD [ "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000" ]


