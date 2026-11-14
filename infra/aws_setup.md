# AWS Deployment Guide

## Prerequisites

- AWS account with appropriate permissions
- AWS CLI configured
- Docker installed
- GitHub repository with secrets configured

## Cost Control

**IMPORTANT**: Set up billing alerts first!

```bash
# Create budget alert
aws budgets create-budget --account-id YOUR_ACCOUNT_ID --budget file://budget.json
```

Recommended budget: $20/month for demo period.

## Step 1: Create ECR Repository

```bash
aws ecr create-repository \
    --repository-name roadscan \
    --image-scanning-configuration scanOnPush=true
```

## Step 2: Create S3 Bucket for DVC

```bash
aws s3 mb s3://roadscan-dvc-YOUR_UNIQUE_ID
aws s3api put-bucket-versioning \
    --bucket roadscan-dvc-YOUR_UNIQUE_ID \
    --versioning-configuration Status=Enabled
```

## Step 3: Create IAM Roles

### ECS Task Execution Role

```bash
aws iam create-role \
    --role-name ecsTaskExecutionRole \
    --assume-role-policy-document file://trust-policy.json
```

### Task Role (for SSM Parameter access)

```bash
aws iam create-role \
    --role-name roadscan-task-role \
    --assume-role-policy-document file://trust-policy.json
```

## Step 4: Store Secrets in SSM

```bash
aws ssm put-parameter \
    --name /roadscan/llm-api-key \
    --value "YOUR_API_KEY" \
    --type SecureString
```

## Step 5: Create ECS Cluster

```bash
aws ecs create-cluster --cluster-name roadscan-cluster
```

## Step 6: Create CloudWatch Log Group

```bash
aws logs create-log-group --log-group-name /ecs/roadscan
```

## Step 7: Configure GitHub Actions

Add secrets to your GitHub repository:

- `AWS_ROLE_ARN`: ARN of the IAM role for GitHub OIDC

## Step 8: Deploy

Push to main branch to trigger deployment:

```bash
git push origin main
```

Or manually trigger via GitHub Actions.

## Cleanup (IMPORTANT!)

After demo period, delete resources:

```bash
# Stop ECS service
aws ecs update-service --cluster roadscan-cluster --service roadscan-service --desired-count 0

# Delete load balancer (if created)
aws elbv2 delete-load-balancer --load-balancer-arn YOUR_ALB_ARN

# Delete ECR images
aws ecr batch-delete-image --repository-name roadscan --image-ids imageTag=latest

# Optional: Delete entire repository
# aws ecr delete-repository --repository-name roadscan --force
```

## Estimated Costs

| Service | Monthly Cost (estimated) |
|---------|-------------------------|
| ECS Fargate (1 vCPU, 2GB, 720h) | ~$30 |
| Application Load Balancer | ~$20 |
| CloudWatch Logs | ~$1 |
| S3 (DVC storage) | ~$1 |
| **Total** | **~$52/month** |

**For demo**: Run only during interview period, keep service at 0 desired count otherwise.

## Security Notes

- Never commit AWS credentials to git
- Use IAM roles, not access keys
- Enable MFA on root account
- Use least-privilege permissions