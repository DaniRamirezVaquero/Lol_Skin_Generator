FROM python:3.12
WORKDIR /app
COPY ../requirements.txt /app/
RUN pip install -r requirements.txt
COPY .. /app/
EXPOSE 8501
ENTRYPOINT [ "streamlit", "run", "streamlit_app.py"]