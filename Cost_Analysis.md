# Azure ML Cost Analysis

## Data Storage Costs
- **Data Storage**:
  - **Description**: Cost for storing data in Azure Blob Storage or Data Lake. (2 cents per GB) We want we only use 5 GB but we want to have sapce for the user to be able to upload data as well. So we will want 100 GB.
  - **Tools**: Azure Blob Storage
  - **Estimated Monthly Cost for Data Storage**: $1.81

    <img src="Storage.png" alt="Data Transfer Costs" width="400" />

  - **Estimated Monthly Cost for Data Transfers**: $1.05

   <img src="options_data_transfer.png" alt="Storage Costs" width="400" />

**Total Data Storage Costs:** $2.85


## 3. Model Training Costs
- **Compute Costs for Training**:
  - **Description**: Compute resources used for training machine learning models. Training takes 30 hours (0.25 per hour). We are using this instance: D16s v3 (16 Core(s), 64 GB RAM).
  - **Tools**: Azure Machine Learning Compute Clusters, GPUs, etc.
  - **Estimated Monthly Cost**: $30.18

  <img src="Azure_ML.png" alt="Storage Costs" width="400" />

**Total Software and Tools Costs:** $30.18

## 4. Model Deployment Cost
- **Deployment Infrastructure**:
  - **Description**: Resources required for deploying the model to production. We ran our container instance with 1 container group, 4GB memory and with 2 cCPU.
  - **Tools**: Azure Container Instances (ACI).
  - **Estimated Monthly Cost**: $1.58

<img src="Azure_Container_instances.png" alt="Data Transfer Costs" width="400" />

**Total Deployment Costs:** $1.58

---
## Total Estimated Monthly Costs:
**Total Data Storage Costs:** $2.85

**Total Software and Tools Costs:** $30.18

**Estimated Monthly Cost:** $1.58

# Total Estimated Monthly Cost: $34.61
---

### Cost Optimization Strategies

#### Data Storage Costs
1. **Optimize Data Usage**: Regularly review and clean up unused or outdated data to minimize storage needs.
2. **Use Blob Storage Tiers**: Utilize Azure Blob Storage's Cool and Archive tiers for infrequently accessed data to save costs.
3. **Data Compression**: Compress data before storing to reduce storage space and costs.

#### Model Training Costs
1. **Optimize Compute Resources**: Fine-tune your training workflows to use the most cost-effective instance types and sizes based on workload requirements.
3. **Parallelize Training**: Distribute training across multiple nodes or GPUs to decrease training time and overall costs.

#### Model Deployment Costs
1. **Right-Size Containers**: Adjust container sizes based on actual workload requirements to avoid over-provisioning.
3. **Auto Scaling**: Implement auto-scaling for container instances based on demand to optimize resource utilization and costs.