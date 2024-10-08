# Use the official Python image from Docker Hub
FROM python:3.11.5

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app's source code to the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000"]