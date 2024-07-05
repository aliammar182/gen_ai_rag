
# Flask Application Deployment on AWS ECS Fargate

This repository contains a Flask application that can be deployed on AWS ECS Fargate with HTTPS enabled. This guide will walk you through the steps necessary to build, push, and redeploy your Flask application using Docker, Amazon ECR, ECS, and an Application Load Balancer with HTTPS.

## Prerequisites

- AWS CLI installed and configured with necessary permissions
- Docker installed and running
- An AWS account with necessary permissions
- Existing ECS Cluster and Task Definition
- Amazon ECR repository for your Docker images

## Steps to Redeploy the Application

### 1. Build and Push the Updated Docker Image to Amazon ECR

1. **Build the Updated Docker Image**:
   Navigate to the directory containing your Dockerfile and build the updated Docker image.
   ```sh
   docker build -t flask-app .
   ```

2. **Tag the Updated Docker Image**:
   Tag the newly built Docker image.
   ```sh
   docker tag flask-app:latest <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest
   ```

3. **Authenticate Docker to Amazon ECR**:
   Authenticate your Docker CLI to your Amazon ECR registry.
   ```sh
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.us-east-1.amazonaws.com
   ```

4. **Push the Updated Docker Image to Amazon ECR**:
   Push the updated Docker image to your Amazon ECR repository.
   ```sh
   docker push <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest
   ```

### 2. Update the ECS Task Definition

1. **Update Task Definition JSON**:
   Ensure your `task-definition.json` is up-to-date with the necessary configurations. Here's an example:
   ```json
   {
       "family": "flask-task",
       "networkMode": "awsvpc",
       "executionRoleArn": "arn:aws:iam::<your-account-id>:role/ecsTaskExecutionRole",
       "taskRoleArn": "arn:aws:iam::<your-account-id>:role/ecsTaskRole",
       "containerDefinitions": [
           {
               "name": "flask-container",
               "image": "<your-account-id>.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest",
               "essential": true,
               "portMappings": [
                   {
                       "containerPort": 5000,
                       "protocol": "tcp"
                   }
               ],
               "environment": [
                   {
                       "name": "S3_BUCKET_NAME",
                       "value": "<your-bucket-name>"
                   },
                   {
                       "name": "PINECONE_API_KEY",
                       "value": "<your-pinecone-api-key>"
                   },
                   {
                       "name": "OPENAI_API_KEY",
                       "value": "<your-openai-api-key>"
                   }
               ],
               "logConfiguration": {
                   "logDriver": "awslogs",
                   "options": {
                       "awslogs-group": "/ecs/flask-app",
                       "awslogs-region": "us-east-1",
                       "awslogs-stream-prefix": "ecs"
                   }
               }
           }
       ],
       "requiresCompatibilities": [
           "FARGATE"
       ],
       "cpu": "256",
       "memory": "512"
   }
   ```

2. **Register the Updated Task Definition**:
   Register the updated task definition with ECS.
   ```sh
   aws ecs register-task-definition --cli-input-json file://task-definition.json
   ```

### 3. Update the ECS Service to Use the New Task Definition

1. **Update the ECS Service**:
   Force a new deployment of the ECS service to use the updated task definition.
   ```sh
   aws ecs update-service      --cluster flask-cluster      --service genai-app      --task-definition flask-task      --force-new-deployment
   ```

### 4. Verify the Deployment

1. **Check ECS Task Logs**:
   Ensure there are no errors related to network or IAM permissions in the ECS task logs.
   ```sh
   aws logs describe-log-streams --log-group-name /ecs/flask-app
   aws logs get-log-events --log-group-name /ecs/flask-app --log-stream-name <log-stream-name>
   ```

2. **Check Load Balancer Health Checks**:
   Verify that the target group health checks are configured correctly and targets are healthy.
   ```sh
   aws elbv2 describe-target-health --target-group-arn <your-target-group-arn>
   ```

### Summary

1. **Build the updated Docker image**.
2. **Tag the updated Docker image**.
3. **Push the updated Docker image to Amazon ECR**.
4. **Update and register the ECS task definition**.
5. **Update the ECS service to use the new task definition**.
6. **Verify the deployment and ensure everything is working correctly**.

By following these steps, you can redeploy your updated Flask application on AWS ECS Fargate. If you encounter any issues or need further assistance, please open an issue in this repository.
