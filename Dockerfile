# pull python base image
FROM python:3.10

# specify working directory
WORKDIR /survival_prediction_api_new

ADD . .

# update pip
RUN pip install --upgrade pip

#Install gradio
RUN pip install gradio



# install dependencies
RUN pip install -r requirements.txt



# copy application files
ADD /app/* ./app/

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]
