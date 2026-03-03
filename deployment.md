# Deployment Guide: Multi-Agent Research System on Azure

This document provides step-by-step instructions for deploying the Multi-Agent Research System as an **Azure Container App (ACA)**. This architecture provides effortless autoscaling (including scaling to zero when idle), integrated secret management via Key Vault, and production-grade distributed memory.

---

## 🏗️ Architecture Overview

The system consists of the following Azure resources:

1.  **Azure Container App (Scaling Workhorse)**: Containerized Python agent logic.
2.  **Azure Container Registry (ACR)**: Private repository for the app’s Docker images.
3.  **Azure Cosmos DB (MongoDB API)**: Durable, globally distributed, long-term/episodic memory.
4.  **Azure Cache for Redis**: Ultra-low-latency cache for short-term working memory.
5.  **Azure Key Vault**: Secure store for secrets (API keys, connection strings).
6.  **Managed Identity**: Secure "passwordless" authentication between all resources.

---

## 🛠️ Prerequisites

- **Azure CLI**: [Install here](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
- **Docker Desktop**: [Install here](https://www.docker.com/products/docker-desktop/) (Used to build and push the image)
- **OpenAI API Key**: Required for "Live" mode.

---

## 🚀 Deployment Steps (Azure CLI)

### 1. Initial Setup & Resource Group
Set your variables and log in:

```bash
# Variables
export RG="rg-stateful-agents"
export LOCATION="eastus"
export ACR_NAME="acragents$(date +%s)" # Must be unique
export ENV_NAME="env-stateful-agents"
export APP_NAME="agent-app"
export KEYVault_NAME="kv-agents-$(date +%s)"
export COSMOS_NAME="cosmos-agents-$(date +%s)"
export REDIS_NAME="redis-agents-$(date +%s)"

# Login and create resource group
az login
az group create --name $RG --location $LOCATION
```

### 2. Infrastructure Provisioning

#### A. Create the Memory Backends (Redis & Cosmos)
Provision the distributed stores:

```bash
# Cosmos DB (MongoDB API) — This may take a few minutes
az cosmosdb create --name $COSMOS_NAME --resource-group $RG --kind MongoDB --locations regionName=$LOCATION failoverPriority=0 isZoneRedundant=False

# Azure Cache for Redis
az redis create --name $REDIS_NAME --resource-group $RG --location $LOCATION --sku Basic --vm-size c0
```

#### B. Setup Secret Management (Key Vault)
Durable security for your keys:

```bash
# Create Key Vault
az keyvault create --name $KEYVault_NAME --resource-group $RG --location $LOCATION

# Add OpenAI API Key to Key Vault
az keyvault secret set --vault-name $KEYVault_NAME --name "OPENAI-API-KEY" --value "your-openai-api-key-here"
```

### 3. Build & Push Container Image

#### A. Create Container Registry (ACR)
Private storage for your code:

```bash
# Create ACR
az acr create --resource-group $RG --name $ACR_NAME --sku Basic

# Log in to ACR
az acr login --name $ACR_NAME
```

#### B. Build and Push 
Leverage the `Dockerfile` in the root:

```bash
# Build the image via ACR (no local Docker engine required)
az acr build --registry $ACR_NAME --image $APP_NAME:latest .
```

### 4. Create the Container App Environment

Azure Container Apps (ACA) manage the networking, scaling, and DAPR integration:

```bash
# Create ACA Environment
az containerapp env create --name $ENV_NAME --resource-group $RG --location $LOCATION
```

### 5. Deploy the Application with Autoscaling

The following command deploys the app, configures a **Managed Identity**, and links it to Key Vault:

```bash
# Create the Container App with scaling rules (CPULoad >= 50%)
az containerapp create \
  --name $APP_NAME \
  --resource-group $RG \
  --environment $ENV_NAME \
  --image "$ACR_NAME.azurecr.io/$APP_NAME:latest" \
  --registry-server "$ACR_NAME.azurecr.io" \
  --system-assigned-identity \
  --secrets "openai-api-key=keyvaultref:https://$KEYVault_NAME.vault.azure.net/secrets/OPENAI-API-KEY" \
  --env-vars \
    "OPENAI_API_KEY=secretref:openai-api-key" \
    "AZURE_REDIS_HOST=$REDIS_NAME.redis.cache.windows.net" \
    "AZURE_COSMOS_DATABASE=agent_memory" \
  --min-replicas 0 \
  --max-replicas 5 \
  --cpu 0.5 --memory 1.0Gi \
  --scale-rule-name "cpu-scaling" \
  --scale-rule-type "cpu" \
  --scale-rule-metadata "concurrentRequests=10"
```

---

## 📈 Autoscaling & Operations

### Performance Tuning
- **Scale-to-Zero**: Setting `--min-replicas 0` ensures you only pay when the agent is actively processing.
- **KEDA Scaling**: ACA uses KEDA under the hood. You can add complex scaling rules (e.g., scale based on message depth in a Queue or Service Bus) via the Azure portal or CLI.

### Monitoring Logs
View real-time logs for your agents:

```bash
az containerapp logs show -n $APP_NAME -g $RG --follow
```

---

## 🧹 Cleanup (Avoid Charges)

When finished testing, delete the resource group to stop all billable services:

```bash
az group delete --name $RG --yes --no-wait
```
