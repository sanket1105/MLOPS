# FROM python:3.8-alpine
# COPY . /app
# WORKDIR /app
# RUN pip install -r requirements.txt
# CMD python app.py


FROM python:3.8-alpine
WORKDIR /code

# Set environment variables (no spaces around '=')
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY . /code
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 from the container
EXPOSE 5000

# Correct CMD syntax (use double quotes for JSON array)
CMD ["flask", "run"]
