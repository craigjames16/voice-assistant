#!/bin/bash
# Login to ECR
aws ecr get-login-password --region ca-central-1 | docker login --username AWS --password-stdin 182491688958.dkr.ecr.ca-central-1.amazonaws.com

# Build and tag
docker build -t voice-assistant .
docker tag voice-assistant:latest 182491688958.dkr.ecr.ca-central-1.amazonaws.com/voice-assistant:latest

# Push to ECR
docker push 182491688958.dkr.ecr.ca-central-1.amazonaws.com/voice-assistant:latest