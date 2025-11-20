import os
import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_recall,
    context_precision,
)
from ragas.run_config import RunConfig
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama 

import langchain

# Environment variables
BASE_URL = os.environ.get("BASE_URL", "")
RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000/")

def get_rag_response(query: str, type: str) -> dict:
    """
    Queries the RAG API and returns the response.
    """
    try:
        response = requests.post(RAG_API_URL+ type, json={"q": query})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying RAG API: {e}")
        return []

def eval(type: str):
    eval_data = [
        {
            "question": "What is the lifespan of a cluster registration token after successful use?",
            "ground_truth": "The Fleet agent is designed to forget the cluster registration token after successful registration. If the cluster needs to be re-registered, a new token must be generated.",
        },
        {
            "question": "What specific Kubernetes Secret type is required for OCI bundle storage configuration?",
            "ground_truth": "The secret must explicitly have the type: fleet.cattle.io/bundle-oci-storage/v1alpha1. Fleet rejects secrets with a different type when enabling OCI storage.",
        },
        {
            "question": "What are the core RBAC privileges of the cluster agent's service account?",
            "ground_truth": "The service account only has privileges to list BundleDeployment resources in its dedicated namespace and to update the status subresource of its BundleDeployment and the Cluster resource.",
        },
        {
            "question": "How is a content patch defined versus a full replacement when using raw YAML overlays?",
            "ground_truth": "A file named `foo` in the overlay folder will replace the file called `foo`. To apply a patch, the overlay file must use the convention of adding `_patch.` to the filename (e.g., `deployment_patch.yaml` patches `deployment.yaml`).",
        },
        {
            "question": "What happens when a GitRepo resource is set to 'paused: true'?",
            "ground_truth": "Changes in Git are not propagated to clusters, and resources are instead marked as OutOfSync. This affects controller operations (creation/update) but not agent operations.",
        },
        {
            "question": "What are the two primary components of Fleet and how do they generally interact?",
            "ground_truth": "The two primary components are the Fleet controller and the cluster agents. They operate in a two-stage pull model: the Fleet controller pulls from git, and the cluster agents pull from the Fleet controller."
        },
        {
            "question": "When configuring GitRepo targets, what does a selector value of {} mean?",
            "ground_truth": "For both GitRepo targets and fleet.yaml targetCustomizations, the value {} for a selector means 'match everything'."
        },
        {
            "question": "In the manager-initiated registration flow, what key network requirement must the Fleet Manager satisfy during the registration phase?",
            "ground_truth": "The Fleet Manager must be able to communicate directly with the downstream cluster API server during the registration phase."
        },
        {
            "question": "What privileges are granted to the service accounts given to managed downstream clusters?",
            "ground_truth": "The service accounts only have privileges to list BundleDeployment in the namespace created specifically for that cluster . They can also update the status subresource of BundleDeployment and the status subresource of the Cluster resource."
        },
        {
            "question": "What is the fundamental deployment unit used in Fleet, and what source contents can it contain?",
            "ground_truth": "The fundamental deployment unit is the **Bundle**. The contents of a Bundle may be Kubernetes manifests, Kustomize configuration, or Helm charts."
        },
        {
            "question": "What is the size limitation for resources deployed from a Git repository through Fleet?",
            "ground_truth": "The scanned resources saved as a resource in Kubernetes must **gzip to less than 1MB**."
        },
        {
            "question": "How can users instruct Fleet to ignore modifications to objects that are amended at runtime?",
            "ground_truth": "Users can specify a custom `jsonPointer` patch within Fleet bundles to instruct Fleet to ignore object modifications or entire objects. This is defined using `diff.comparePatches` in the `fleet.yaml`."
        },
        {
            "question": "What is the precedence order for applying changes to `values.yaml` from multiple sources in a Helm bundle?",
            "ground_truth": "When changes are applied from multiple sources simultaneously, they update in the following order: 1. `helm.values`, 2. `helm.valuesFiles`, 3. `helm.valuesFrom`. This means `valuesFrom` always overrides both `valuesFiles` and `values`."
        },
        {
            "question": "What is HelmOps, and what is its primary advantage over using a GitRepo resource?",
            "ground_truth": "HelmOps is a simplified way of creating bundles by directly pointing to a Helm repository or an OCI registry, without needing to set up a git repository. It relies on a Helm registry as its source of truth, unlike GitOps which uses a Git repository."
        },
        {
            "question": "What customization method can be used in raw Kubernetes YAML deployments for overlaying or patching resources?",
            "ground_truth": "If raw YAML is used (not Kustomize or Helm), the `overlays/` folder is treated specially. The content of the overlay resources uses a file name based approach where a file will replace a file with the same name, or a file name ending in `_patch.` will patch the content of the base file."
        },
        {
            "question": "When manually configuring rollout partitions, what is the significance of the partition order in the `fleet.yaml` file?",
            "ground_truth": "Fleet processes partitions in the order they appear in the `fleet.yaml` file."
        },
        {
            "question": "What default rollout setting causes partitions to be treated as Ready immediately, regardless of actual workload status?",
            "ground_truth": "By default, Fleet sets `maxUnavailable` to 100%, meaning all clusters in a partition can be NotReady and still be considered Ready, which causes Fleet to proceed through all partitions regardless of actual readiness."
        },
        {
            "question": "What is the primary role of the `ClusterRegistrationToken`?",
            "ground_truth": "The cluster registration token is a credential that authorizes the downstream cluster agent to be able to initiate the registration process. Internally, it is managed by creating Kubernetes service accounts that have permissions to create ClusterRegistrationRequests within a specific namespace."
        },
        {
            "question": "What happens if an OCI storage secret is missing or invalid when OCI storage is enabled for bundles?",
            "ground_truth": "Fleet does not fall back to etcd if the secret is missing or invalid; instead, it logs an error and skips the deployment."
        },
         {
            "question": "Describe the security roles and connectivity requirements for both agent-initiated and manager-initiated cluster registration flows.",
            "ground_truth": "In the **agent-initiated registration** flow, the downstream cluster installs an agent using a cluster registration token, and the registration communication flows from the managed cluster to the Fleet Controller (upstream). The Fleet Manager does not need direct inbound network access to the downstream cluster API. In the **manager-initiated registration** flow, the Fleet Manager initiates the registration by deploying the agent to the downstream cluster. This style requires the Fleet Manager to be able to communicate directly with the downstream cluster API server during the registration phase, although no further contact is needed after registration. For both styles, the service accounts eventually granted to the clusters only have privileges to list `BundleDeployment` in their cluster-specific namespace and update the `status` subresource of `BundleDeployment` and the `Cluster` resource."
        },
        {
            "question": "What is the primary deployment unit in Fleet, and what three types of source content can it contain? Additionally, explain how a GitRepo resource relates to this deployment unit during Fleet's scanning process.",
            "ground_truth": "The primary deployment unit in Fleet is the **Bundle**. The contents of a Bundle may be **Kubernetes manifests**, **Kustomize configuration**, or **Helm charts**. When a `GitRepo` resource is scanned, Fleet fetches the resources from the Git repository, and the scanning process produces one or more Bundles. Each scanned path in the Git repository is internally managed as an independent Bundle ."
        },
        {
            "question": "When configuring rollouts using partitions in a `fleet.yaml`, what three default values are used for `maxUnavailable`, `maxUnavailablePartitions`, and `autoPartitionSize`, and what is the consequence of the default `maxUnavailable` setting?",
            "ground_truth": "The default values for rollout configuration are: `maxUnavailable` defaults to **100%**, `maxUnavailablePartitions` defaults to **0** , and `autoPartitionSize` defaults to **25%**. The consequence of `maxUnavailable` being set to **100%** by default is that all clusters in a partition can be `NotReady` and still be considered `Ready`, which causes Fleet to proceed through all partitions regardless of actual workload readiness."
        },
        {
            "question": "Analyze the following excerpt from a `fleet.yaml` regarding cluster selection and modification rules. Which cluster(s) would receive the `db-replication: 'true'` modification, and what is the precedence order for applying changes to `values.yaml` if this customization block also included `helm.values` and `helm.valuesFiles`?",
            "ground_truth": "The `targetCustomizations` are evaluated in order, and the first one to match is used for that cluster. The first customization block targets clusters matching the selector `env: test`. The second customization block targets clusters matching `env: prod`. Therefore, only clusters labeled with **`env: test`** and **`env: prod`** would receive the customization `db-replication: 'true'`. When changes are applied to `values.yaml` from multiple sources at the same time, the values update in the following precedence order (from lowest to highest override priority): 1. `helm.values`, 2. `helm.valuesFiles`, 3. `helm.valuesFrom`."
        },
        {
            "question": "If an operator wants to use a webhook instead of polling for a GitRepo, outline the three key configuration steps required, and explain what happens to the polling interval once the webhook is configured.",
            "ground_truth": "The three key configuration steps required for using webhooks instead of polling are: 1. **Configure the webhook service** by creating an ingress that points to the `gitjob` service. 2. **Configure the webhook callback URL** in the webhook provider's settings (e.g., GitHub). 3. **(Optional but recommended)** **Configure a webhook secret** in Kubernetes (either a global secret named `gitjob-webhook` in `cattle-fleet-system` or a specific secret referenced in the GitRepo) to validate the webhook payload. Once the webhook is configured, the **polling interval will be automatically adjusted to 1 hour**."
        }
    ]

    for item in eval_data:
        rag_response = get_rag_response(item["question"], type)
        item["contexts"] = [doc["page_content"] for doc in rag_response]

    dataset = Dataset.from_list(eval_data)
    llm = ChatOllama(model="gpt-oss:20b", base_url=BASE_URL)
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b", base_url=BASE_URL)
    metrics = [
        context_recall,
        context_precision,
    ]

    print(f"Running Ragas evaluation for {type}...")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        batch_size=1
    )

    print(f"Ragas evaluation results for {type}: {result}")

    df = result.to_pandas()
    print(df)
    df.to_json(f'eval_{type}.json', orient='records')

def main():
    eval("recursive")
    eval("markdown")
    eval("hierarchical")
    eval("summary")

if __name__ == "__main__":
    main()
