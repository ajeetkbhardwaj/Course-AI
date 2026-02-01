General Machine Learning Workflow

1. Source Data and Preprocess Data
2. Design Model
3. Train and Evaluate Model
4. Test Model
5. Deploy Model
6. Monitor and Maintain

Tasks like review each

In ML lifecycle, the organizations faces operational bottlenecks during taking models from experimentation to production.

1. Source and Preparation of Data
   There are certain tasks we have to performs like

- Collect raw data from databases, APIs, sensors, logs, etc.
- Clean, label, and preprocess (handle missing values, normalization, encoding, etc.).
- Split into train/validation/test datasets
  using
  Tools like pandas, numpy, Apache Spark, DVC and Airflow etc. But the main challenges still remains to ensure the
- Data quality, bias, schema drift and reproducibility etc

2. Design Model
   There are certain tasks like

- Which model we should choose like linear(regression, logistic etc), nonlinear(tree, svm etc), ensamble(AdaBoost, Random Forest etc) If we have same dataset.
- CNN-based DNN architecture, RNN-based architecture or Transformer based architecture for our problem statement and datasource.
- Then define the architecture or algorithm and its hyperparameters
- At the end, we needed to implement it using frameworks like TensorFlow, PyTorch or Scikit-learn.

But during handling those tasks we still face the challenges to handle like framework compatibility, architecture search and reproducebility of results accross enviroments etc.

3. Train and Evaluate Model
   Tasks:

- Train on shared or distributed hardware (GPUs, TPUs, or clusters).
- Monitor loss, metrics, and overfitting.
- Evaluate using validation/test data.

Tools: PyTorch Lightning, MLflow, TensorBoard, Ray Tune

Challenges:

- Shared hardware access across teams
- Framework interoperability
- Dependency version conflicts

4. Test the model
   Tasks:

- Validate performance on unseen or real-world-like data.
- Perform stress tests and adversarial testing.
- Verify robustness, explainability, and fairness.

Tools: SHAP, LIME, Alibi, robustness libraries

Challenges: Dataset drift, reproducibility, and environment mismatch.

5. Deploy the Model

Tasks:

- Package and containerize model (Docker, ONNX, TorchScript).
- Deploy via REST API, batch jobs, or streaming pipelines.
- Integrate with CI/CD pipelines for continuous updates.
  Tools: Kubernetes, Docker, FastAPI, BentoML, MLflow, Kubeflow
  Challenges:
- Managing software dependencies and runtime environments
- Hardware accessibility across teams and frameworks
- Deployment/versioning control — rollback, A/B testing, canary releases

6. Monitor and Maintain
   Tasks:

- Monitor model drift, performance degradation, and latency.
- Retrain periodically as data evolves.
- Track versions and metrics using MLOps platforms.
  Tools: MLflow, EvidentlyAI, Prometheus, Grafana
  Challenges: Automating retraining, detecting drift, and ensuring auditability.

Remarks - Around 40–50% of companies report it takes over a month to move a model from development to production (source: State of MLOps Reports by Algorithmia, 2023; McKinsey AI Report).

The main blockers challenges are

1. Dependency and environment complexity
2. Limited access to uniform compute environments
3. Lack of deployment automation and model version control


Dependecies conflict problem in ML environment

1. how to connects to hardware, OS, CUDA and framework-level dependencies and then possible modern solutions that make multi-project ML work feasible.

**Possible Senario for dependencies problems**

| Component                | Project 1             | Project 2             |
| ------------------------ | --------------------- | --------------------- |
| **Framework**      | TensorFlow            | PyTorch               |
| **Python version** | 3.8                   | 3.7                   |
| **CUDA toolkit**   | 10.x                  | 11.x                  |
| **Hardware**       | Same GPU              | Same GPU              |
| **OS**             | Shared (e.g., Ubuntu) | Shared (e.g., Ubuntu) |

Possible conflicts might be

* **Python version mismatch**

  * You cannot run both Python 3.7 and 3.8 in the same environment natively.
* **CUDA toolkit incompatibility**

  * TensorFlow (CUDA 10) and PyTorch (CUDA 11) require different GPU drivers and libraries.
* **Framework version conflicts**

  * TensorFlow and PyTorch depend on specific versions of `numpy`, `protobuf`, `grpc`, etc.
* **Hardware contention**

  * Both projects need GPU resources (which cannot be shared easily across incompatible CUDA stacks).
* **Result:**

  * You can **only run one project at a time** in the same environment.
  * Installing one breaks the other

We know that ML Stacks are deeply coupled so, the possible root cause would be...


```plaintext
Hardware (GPU/CPU)
   ↓
Operating System (Linux/Windows)
   ↓
NVIDIA Drivers + CUDA Toolkit + cuDNN
   ↓
Deep Learning Framework (TensorFlow / PyTorch)
   ↓
Python Environment and Packages
   ↓
Your Project Code

```

Even small mismatches (e.g., CUDA 10 vs 11) break compatibility across this stack. Each layer depends on the versions below it. 

1. Vertual Environment : We can use the vertualenv or the conda env to isolate the python and package dependencies.
```bash
# Project 1
conda create -n tf38 python=3.8 tensorflow cudatoolkit=10.1
conda activate tf38

# Project 2
conda create -n torch37 python=3.7 pytorch cudatoolkit=11.0 -c pytorch
conda activate torch37

```
But Still they share the same host CUDA drivers → may conflict if framework versions mismatch significantly.

2. Docker/Containerization
We use Docker to isolate entire enviroments including CUDA Versions and drivers. Where each project gets its own container image.

```paintext
# Example Dockerfiles
# Project 1
FROM tensorflow/tensorflow:2.8.0-gpu
RUN pip install -r requirements.txt

# Project 2
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
RUN pip install -r requirements.txt

```
Then we run the following command
```bash
docker run --gpus all -it project1:latest
docker run --gpus all -it project2:latest
 
```

MLOps Practices/Modern Solutions

| Problem                   | Solution                                                                               |
| ------------------------- | -------------------------------------------------------------------------------------- |
| Dependency chaos          | Use Docker, Conda, or poetry for environment reproducibility                           |
| Shared hardware conflicts | Use cloud ML platforms (Vertex AI, SageMaker, Azure ML) with containerized jobs        |
| Deployment/versioning     | Use MLflow, DVC, or Kubeflow Pipelines for model tracking and reproducible deployments |
| Long deployment cycles    | Automate CI/CD with GitHub Actions + model registry integration                        |


What are the possible benifits of using docker ?
Complete isolation (framework + CUDA + Python)
Multiple versions coexist
Reproducible builds
Portable between systems and teams

Vertual Machines or Cloud Enviroments
If hardware conflict persists, spin up seperate GPU VMs on platforms like AWS EC2 (Deep Learning AMIs),
Google Vertex AI,
Azure ML,
Paperspace, RunPod, or Lambda Labs. Wheres
Each VM runs its own environment, driver, and dependencies

We can also use the Environment Management Tools like
1. Poetry or pipenv for dependency locking
2. Conda-lock for reproducible environment YAMLs
3. Docker Compose to manage multiple project containers
4. NVIDIA Container Toolkit for GPU passthrough in Docker